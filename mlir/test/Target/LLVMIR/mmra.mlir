// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @native
// CHECK: load
// CHECK-SAME: !mmra ![[MMRA0:[0-9]+]]
// CHECK: fence
// CHECK-SAME: !mmra ![[MMRA1:[0-9]+]]
// CHECK: store {{.*}}, align 4{{$}}

#mmra_tag = #llvm.mmra_tag<"foo":"bar">

llvm.func @native(%x: !llvm.ptr, %y: !llvm.ptr) {
  %0 = llvm.load %x {llvm.mmra = #mmra_tag} : !llvm.ptr -> i32
  llvm.fence syncscope("workgroup-one-as") release
    {llvm.mmra = [#llvm.mmra_tag<"amdgpu-synchronize-as":"local">, #mmra_tag]}
  llvm.store %0, %y {llvm.mmra = []} : i32, !llvm.ptr
  llvm.return
}

// Actual MMRA metadata
// CHECK-DAG: ![[MMRA0]] = !{!"foo", !"bar"}
// CHECK-DAG: ![[MMRA_PART0:[0-9]+]] = !{!"amdgpu-synchronize-as", !"local"}
// CHECK-DAG: ![[MMRA1]] = !{![[MMRA_PART0]], ![[MMRA0]]}

// -----

// CHECK-LABEL: define void @foreign_op
// CHECK: call void @llvm.amdgcn.load.to.lds
// CHECK-SAME: !mmra ![[MMRA0:[0-9]+]]
llvm.func @foreign_op(%g: !llvm.ptr<1>, %l: !llvm.ptr<3>) {
  rocdl.load.to.lds %g, %l, 4, 0, 0 {llvm.mmra = #llvm.mmra_tag<"fake":"example">} : !llvm.ptr<1>
  llvm.return
}

// CHECK: ![[MMRA0]] = !{!"fake", !"example"}
