#!/bin/sh

# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The mvifdiff.sh script works like the mv(1) command, except
# that it does not touch the destination file if its contents
# are the same as the source file.

if cmp -s "$1" "$2" ; then
  rm "$1"
else
  mv "$1" "$2"
fi
