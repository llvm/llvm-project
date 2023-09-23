#!/bin/sh

set -e
set -x

if test ! "$(dirname $0)" -ef '.'; then
    echo "The script must be executed from its current directory."
    exit 1
fi

if test "$#" -ne 1; then
    echo "Usage: $0 <llvm-config>"
    exit 1
fi

llvm_config=$1
default_mode=
support_static_mode=false
support_shared_mode=false

llvm_config() {
    "$llvm_config" $@
}

if llvm_config --link-static --libs; then
    default_mode=static
    support_static_mode=true
fi

if llvm_config --link-shared --libs; then
    default_mode=shared
    support_shared_mode=true
fi

if test -z "$default_mode"; then
    echo "Something is wrong with the llvm-config command provided."
    exit 1
fi

base_cflags=$(llvm_config --cflags)
ldflags="$(llvm_config --ldflags) -lstdc++ -fPIC"
llvm_targets=$(llvm_config --targets-built)

append_context() {
    context_name=$1
    linking_mode=$2
    echo "(context (default
 (name ${context_name})
 (env
  (_
   (c_flags $base_cflags)
   (env-vars
    (LLVM_CONFIG $llvm_config)
    (LINK_MODE $linking_mode))))))
" >> "dune-workspace"
}

echo "(lang dune 3.2)
" > "dune-workspace"

if $support_shared_mode; then
    append_context shared --link-shared
fi
if $support_static_mode; then
    append_context static --link-static
fi
