#!/bin/sh
# If the file still contains 'vector<4xf32>', it's interesting!
if grep -q 'vector<4xf32>' "$1"; then
  exit 1
else
  exit 0
fi
