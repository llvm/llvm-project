#!/bin/sh
if [ "$(uname)" = "Darwin" ] ; then
    export DYLD_LIBRARY_PATH=.:../..  # for macOS
else
    export LD_LIBRARY_PATH=.:../..    # for everyone else
fi
python runtest.py $@
