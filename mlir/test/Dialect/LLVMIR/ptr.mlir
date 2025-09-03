// RUN: mlir-opt %s --verify-roundtrip

// Check that LLVM ops accept ptr values.
llvm.func @llvm_ops_with_ptr_values(%arg0: !llvm.ptr, %arg1: !llvm.struct<(!ptr.ptr<#llvm.address_space<3>>)>) {
  %1 = llvm.load %arg0 : !llvm.ptr -> !ptr.ptr<#llvm.address_space<1>>
  llvm.store %1, %arg0 : !ptr.ptr<#llvm.address_space<1>>, !llvm.ptr
  llvm.store %arg1, %arg0 : !llvm.struct<(!ptr.ptr<#llvm.address_space<3>>)>, !llvm.ptr
  llvm.return
}
