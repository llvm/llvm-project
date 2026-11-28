// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define i1 @tensormap_query_alloc_class_large_generic(ptr %0) {
llvm.func @tensormap_query_alloc_class_large_generic(%addr : !llvm.ptr) -> i1 {
  // CHECK-NEXT: %2 = call i1 @llvm.nvvm.tensormap.query.alloc.class.large.p0(ptr %0)
  %res = nvvm.tensormap.query.alloc_class_large %addr : !llvm.ptr -> i1

  // CHECK-NEXT: ret i1 %2
  llvm.return %res : i1
  // CHECK-NEXT: }
}

// CHECK-LABEL: define i1 @tensormap_query_alloc_class_large_global(ptr addrspace(1) %0) {
llvm.func @tensormap_query_alloc_class_large_global(%addr : !llvm.ptr<1>) -> i1 {
  // CHECK-NEXT: %2 = call i1 @llvm.nvvm.tensormap.query.alloc.class.large.p1(ptr addrspace(1) %0)
  %res = nvvm.tensormap.query.alloc_class_large %addr : !llvm.ptr<1> -> i1

  // CHECK-NEXT: ret i1 %2
  llvm.return %res : i1
  // CHECK-NEXT: }
}

// CHECK-LABEL: define i1 @tensormap_query_alloc_class_large_shared(ptr addrspace(3) %0) {
llvm.func @tensormap_query_alloc_class_large_shared(%addr : !llvm.ptr<3>) -> i1 {
  // CHECK-NEXT: %2 = call i1 @llvm.nvvm.tensormap.query.alloc.class.large.p3(ptr addrspace(3) %0)
  %res = nvvm.tensormap.query.alloc_class_large %addr : !llvm.ptr<3> -> i1

  // CHECK-NEXT: ret i1 %2
  llvm.return %res : i1
  // CHECK-NEXT: }
}

// CHECK-LABEL: define i1 @tensormap_query_alloc_class_large_const(ptr addrspace(4) %0) {
llvm.func @tensormap_query_alloc_class_large_const(%addr : !llvm.ptr<4>) -> i1 {
  // CHECK-NEXT: %2 = call i1 @llvm.nvvm.tensormap.query.alloc.class.large.p4(ptr addrspace(4) %0)
  %res = nvvm.tensormap.query.alloc_class_large %addr : !llvm.ptr<4> -> i1

  // CHECK-NEXT: ret i1 %2
  llvm.return %res : i1
  // CHECK-NEXT: }
}

// CHECK-LABEL: define i1 @tensormap_query_alloc_class_large_param(ptr %0) {
llvm.func @tensormap_query_alloc_class_large_param(%addr : !llvm.ptr) -> i1 {
  // CHECK-NEXT: %2 = addrspacecast ptr %0 to ptr addrspace(101)
  // CHECK-NEXT: %3 = call i1 @llvm.nvvm.tensormap.query.alloc.class.large.p101(ptr addrspace(101) %2)
  %res = nvvm.tensormap.query.alloc_class_large in_param_space, %addr : !llvm.ptr -> i1

  // CHECK-NEXT: ret i1 %3
  llvm.return %res : i1
  // CHECK-NEXT: }
}
