// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

func.func @test_stacksave_stackrestore() {
  // expected-remark @below {{found an instance of 'free' on resource 'AutomaticAllocationScope'}}
  %ptr = llvm.intr.stacksave : !llvm.ptr
  // expected-remark @below {{operation has no memory effects}}
  %c1 = llvm.mlir.constant(1 : i32) : i32
  // expected-remark @below {{found an instance of 'allocate' on op result 0, on resource 'AutomaticAllocationScope'}}
  %alloca = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // expected-remark @below {{found an instance of 'free' on resource 'AutomaticAllocationScope'}}
  llvm.intr.stackrestore %ptr : !llvm.ptr
  llvm.return
}
