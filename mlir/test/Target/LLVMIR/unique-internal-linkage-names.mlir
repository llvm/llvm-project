// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @test_(ptr noalias %0)
// CHECK: call void @_QFtestPfoo.__uniq.12345(ptr %0)

llvm.func @test_(%arg0: !llvm.ptr {llvm.noalias}) {
  llvm.call @_QFtestPfoo.__uniq.12345(%arg0) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: define internal void @_QFtestPfoo.__uniq.12345(ptr noalias %0)
llvm.func internal @_QFtestPfoo.__uniq.12345(%arg0: !llvm.ptr {llvm.noalias}) attributes {sample_profile_suffix_elision_policy = "selected"} {
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.add %0, %1 : i32
  llvm.store %2, %arg0 : i32, !llvm.ptr
  llvm.return
}

// CHECK: attributes #[[ATTRS:.*]] = { "sample-profile-suffix-elision-policy"="selected" }
