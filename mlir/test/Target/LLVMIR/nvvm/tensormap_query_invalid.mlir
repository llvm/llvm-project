// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @tensormap_query_alloc_class_large_in_param_space_non_generic(%addr : !llvm.ptr<4>) -> i1 {
  // expected-error @+1 {{in_param_space can only be specified for a generic pointer}}
  %res = nvvm.tensormap.query.alloc_class_large in_param_space, %addr : !llvm.ptr<4> -> i1
  llvm.return %res : i1
}
