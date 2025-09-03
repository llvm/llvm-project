// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: declare ptr @llvm_ptr_address_space(ptr addrspace(1), ptr addrspace(3))
llvm.func @llvm_ptr_address_space(!ptr.ptr<#llvm.address_space<1>>, !ptr.ptr<#llvm.address_space<3>>) -> !ptr.ptr<#llvm.address_space<0>>

// CHECK-LABEL: define void @llvm_ops_with_ptr_values
// CHECK-SAME:   (ptr %[[ARG:.*]]) {
// CHECK-NEXT:   %[[V0:.*]] = load ptr addrspace(1), ptr %[[ARG]], align 8
// CHECK-NEXT:   store ptr addrspace(1) %[[V0]], ptr %[[ARG]], align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @llvm_ops_with_ptr_values(%arg0: !llvm.ptr) {
  %1 = llvm.load %arg0 : !llvm.ptr -> !ptr.ptr<#llvm.address_space<1>>
  llvm.store %1, %arg0 : !ptr.ptr<#llvm.address_space<1>>, !llvm.ptr
  llvm.return
}
