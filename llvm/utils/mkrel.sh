#!/bin/sh

#
# Shell Script: mkrel
#
# Description:
#   Make LLVM Release source tarballs by grabbing the source from the CVS
#   repository.
#
# Usage:
#   mkrel <version> <release tag> <dir>
#

#
# Constants
#
cvsroot=":pserver:anon@llvm-cvs.cs.uiuc.edu:/var/cvs/llvm"

#
# Save the command line arguments into some variables.
#
version=$1
tag=$2
dir=$3

#
# Create the working directory and make it the current directory.
#
mkdir -p $dir
cd $dir

#
# Extract the LLVM sources given the label.
#
cvs -d $cvsroot export -r $tag llvm llvm-gcc

#
# Create source tarballs.
#
tar -cvf - llvm | gzip > llvm-${version}.tar.gz
tar -cvf - llvm-gcc | gzip > cfrontend-${version}.source.tar.gz
