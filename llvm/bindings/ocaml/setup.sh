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

    core_libs=$(llvm_config $linking_mode --libs core support)
    analysis_libs=$(llvm_config $linking_mode --libs analysis)
    bitreader_libs=$(llvm_config $linking_mode --libs bitreader)
    bitwriter_libs=$(llvm_config $linking_mode --libs bitwriter)
    executionengine_libs=$(llvm_config $linking_mode --libs executionengine mcjit native)
    irreader_libs=$(llvm_config $linking_mode --libs irreader)
    transformutils_libs=$(llvm_config $linking_mode --libs transformutils)
    passes_libs=$(llvm_config $linking_mode --libs passes)
    target_libs=$(llvm_config $linking_mode --libs target)
    linker_libs=$(llvm_config $linking_mode --libs linker)
    all_backend_libs=$(llvm_config $linking_mode --libs $llvm_targets)

    echo "(context (default
 (name ${context_name})
 (env
  (_
   (c_flags $base_cflags)
   (env-vars
    (LLVMCore_LIB \"$ldflags $core_libs\")
    (LLVMAnalysis_LIB \"$ldflags $analysis_libs\")
    (LLVMBitReader_LIB \"$ldflags $bitreader_libs\")
    (LLVMBitWriter_LIB \"$ldflags $bitwriter_libs\")
    (LLVMExecutionEngine_LIB \"$ldflags $executionengine_libs\")
    (LLVMIRReader_LIB \"$ldflags $irreader_libs\")
    (LLVMTransformUtils_LIB \"$ldflags $transformutils_libs\")
    (LLVMPasses_LIB \"$ldflags $passes_libs\")
    (LLVMTarget_LIB \"$ldflags $target_libs\")
    (LLVMLinker_LIB \"$ldflags $linker_libs\")
    (LLVMAll_backends_LIB \"$ldflags $all_backend_libs\"))))))
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
