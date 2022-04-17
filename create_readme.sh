#!/usr/bin/env bash

jupyter nbconvert --to markdown "Personalised Sound Meter.ipynb"
pandoc --from=gfm --to=gfm-raw_html --wrap=preserve --output README.md "Personalised Sound Meter.md"
rm "Personalised Sound Meter.md"
