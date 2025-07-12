// RUN: mlir-opt %s -llvm-data-layout-from-target --split-input-file --verify-diagnostics

// expected-error @+1 {{failed to obtain llvm::DataLayout from #llvm.target}}
module attributes { dlti.dl_spec = #dlti.dl_spec<index = 32>,
llvm.target =
    #llvm.target<triple="x64_86-unknown-linux",
                 chip="lakesky"> } {
}
