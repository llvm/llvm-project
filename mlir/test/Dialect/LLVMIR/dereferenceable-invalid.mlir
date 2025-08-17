// RUN: mlir-opt --allow-unregistered-dialect -split-input-file -verify-diagnostics %s

llvm.func @deref(%arg0: !llvm.ptr) {
    // expected-error @below {{op expected op to return a single LLVM pointer type}}
    %0 = llvm.load %arg0 dereferenceable<bytes = 8> {alignment = 8 : i64} : !llvm.ptr -> i64
    llvm.return
}
