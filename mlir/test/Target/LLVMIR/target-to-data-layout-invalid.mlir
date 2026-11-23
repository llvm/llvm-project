// REQUIRES: target=x86{{.*}}
// RUN: mlir-opt %s -llvm-target-to-data-layout --split-input-file --verify-diagnostics

// expected-error @+1 {{failed to obtain llvm::DataLayout for #llvm.target}}
module attributes { dlti.dl_spec = #dlti.dl_spec<index = 32>,
llvm.target =
    #llvm.target<triple="x64_86-unknown-linux",
                 chip="NON-EXISTING CHIP"> } {
}
