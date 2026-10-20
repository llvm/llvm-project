// REQUIRES: target=x86{{.*}}
// RUN: mlir-opt %s -llvm-target-to-data-layout="initialize-llvm-targets=false" --split-input-file --verify-diagnostics

// Without initializing the (right) LLVM targets/backends ("initialize-llvm-targets=false"),
// it is not possible to obtain LLVM's DataLayout for the target.

// expected-error @+1 {{failed to obtain llvm::DataLayout for #llvm.target}}
module attributes { dlti.dl_spec = #dlti.dl_spec<index = 32>,
llvm.target =
    #llvm.target<triple="x64_86-unknown-linux",
                 chip="skylake"> } {
}
