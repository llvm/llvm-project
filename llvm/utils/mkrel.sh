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
echo "Changing directory to $dir"
cd $dir

#
# Extract the LLVM sources given the label.
#
echo "Extracting source $tag from $cvsroot"
cvs -d $cvsroot export -r $tag llvm llvm-gcc

#
# Move the llvm-gcc sources so that they match what is used by end-users.
#
mkdir -p cfrontend
mv llvm-gcc cfrontend/src

#
# Create source tarballs.
#
tar -cf - llvm | gzip > llvm-${version}.tar.gz
tar -cf - cfrontend | gzip > cfrontend-${version}.source.tar.gz
