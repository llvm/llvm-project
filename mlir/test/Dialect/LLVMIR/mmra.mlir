// RUN: mlir-opt %s -split-input-file --verify-roundtrip --mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: llvm.func @native
// CHECK: llvm.load
// CHECK-SAME: llvm.mmra = #llvm.mmra_tag<"foo":"bar">
// CHECK: llvm.fence
// CHECK-SAME: llvm.mmra = [#llvm.mmra_tag<"amdgpu-synchronize-as":"local">, #llvm.mmra_tag<"foo":"bar">]
// CHECK: llvm.store
// CHECK-SAME: llvm.mmra = #llvm.mmra_tag<"foo":"bar">

#mmra_tag = #llvm.mmra_tag<"foo":"bar">

llvm.func @native(%x: !llvm.ptr, %y: !llvm.ptr) {
  %0 = llvm.load %x {llvm.mmra = #mmra_tag} : !llvm.ptr -> i32
  llvm.fence syncscope("workgroup-one-as") release
    {llvm.mmra = [#llvm.mmra_tag<"amdgpu-synchronize-as":"local">, #mmra_tag]}
  llvm.store %0, %y {llvm.mmra = #llvm.mmra_tag<"foo":"bar">} : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @foreign_op
// CHECK: rocdl.load.to.lds
// CHECK-SAME: llvm.mmra = #llvm.mmra_tag<"fake":"example">
llvm.func @foreign_op(%g: !llvm.ptr<1>, %l: !llvm.ptr<3>) {
  rocdl.load.to.lds %g, %l, 4, 0, 0 {llvm.mmra = #llvm.mmra_tag<"fake":"example">} : !llvm.ptr<1>
  llvm.return
}
