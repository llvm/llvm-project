#!/bin/sh

if [ $# != 1 ]; then
  echo "$0 <output path>"
  exit 1
fi

md5cmd=$(which md5sum)
is_md5=0
if [ ! -x "$md5cmd" ]; then
    md5cmd=$(which md5)
    if [ -x "$md5cmd" ]; then
        is_md5=1
    else
        md5cmd=$(which csum)
        if [ ! -x "$md5cmd" ]; then
            echo "error: unable to find either 'md5sum', 'md5' or 'csum'"
            exit 1
        fi
        # Pass options to make csum behave identically to md5sum.
        md5cmd="${md5cmd} -h MD5 -"
    fi
fi

mv $1 $1.bak
if [ $is_md5 = "1" ]; then
    $md5cmd -q < $1.bak > $1
else
    $md5cmd < $1.bak | cut -d' ' -f 1 > $1
fi
rm -f $1.bak
