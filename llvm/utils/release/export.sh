#!/bin/sh
#===-- tag.sh - Tag the LLVM release candidates ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# Create branches and release candidates for the LLVM release.
#
#===------------------------------------------------------------------------===#

set -e

projects="llvm clang test-suite compiler-rt libcxx libcxxabi clang-tools-extra polly lldb lld openmp libunwind flang"

release=""
rc=""

usage() {
    echo "Export the Git sources and build tarballs from them"
    echo "usage: `basename $0`"
    echo " "
    echo "  -release <num> The version number of the release"
    echo "  -rc <num>      The release candidate number"
    echo "  -final         The final tag"
}

export_sources() {
    release_no_dot=`echo $release | sed -e 's,\.,,g'`
    tag="llvmorg-$release"

    if [ "$rc" = "final" ]; then
        rc=""
    else
        tag="$tag-$rc"
    fi

    llvm_src_dir=llvm-project-$release$rc
    mkdir -p $llvm_src_dir

    echo $tag
    echo "Fetching LLVM project source ..."
    curl -L https://github.com/llvm/llvm-project/archive/$tag.tar.gz | \
        tar -C $llvm_src_dir --strip-components=1 -xzf -

    echo "Creating tarball for llvm-project ..."
    tar -cJf llvm-project-$release$rc.tar.xz $llvm_src_dir

    echo "Fetching LLVM test-suite source ..."
    mkdir -p $llvm_src_dir/test-suite
    curl -L https://github.com/llvm/test-suite/archive/$tag.tar.gz | \
        tar -C $llvm_src_dir/test-suite --strip-components=1 -xzf -

    for proj in $projects; do
        echo "Creating tarball for $proj ..."
        mv $llvm_src_dir/$proj $llvm_src_dir/$proj-$release$rc.src
        tar -C $llvm_src_dir -cJf $proj-$release$rc.src.tar.xz $proj-$release$rc.src
    done
}

while [ $# -gt 0 ]; do
    case $1 in
        -release | --release )
            shift
            release=$1
            ;;
        -rc | --rc )
            shift
            rc="rc$1"
            ;;
        -final | --final )
            rc="final"
            ;;
        -h | -help | --help )
            usage
            exit 0
            ;;
        * )
            echo "unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "x$release" = "x" ]; then
    echo "error: need to specify a release version"
    exit 1
fi

# Make sure umask is not overly restrictive.
umask 0022

export_sources
exit 0
