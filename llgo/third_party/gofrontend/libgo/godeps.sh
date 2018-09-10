#!/bin/sh

# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The godeps.sh script outputs a dependency file for a package.  The
# dependency file is then included in the libgo Makefile.  This is
# automatic dependency generation, Go style.

# The first parameter is the name of the file being generated.  The
# remaining parameters are the names of Go files which are scanned for
# imports.

set -e

if test $# = 0; then
    echo 1>&2 "Usage: godeps.sh OUTPUT INPUTS..."
    exit 1
fi

output=$1
shift

deps=`for f in $*; do cat $f; done | 
  sed -n -e '/^import.*"/p; /^import[ 	]*(/,/^)/p' |
  grep '"' |
  grep -v '"unsafe"' |
  sed -e 's/^.*"\([^"]*\)".*$/\1/' -e 's/$/.gox/' |
  sort -u`

echo $output: $deps
