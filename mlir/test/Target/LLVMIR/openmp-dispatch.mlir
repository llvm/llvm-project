// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.mlir.global external @x() {addr_space = 0 : i32} : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

llvm.func @variant1() -> ()
llvm.func @variant2() -> ()

// CHECK-LABEL: define void @foo_dispatch()
llvm.func @foo_dispatch() {
  // CHECK: %[[ADDR:.*]] = load i32, ptr @x
  // CHECK: %[[CMP:.*]] = icmp eq i32 %[[ADDR]], 1
  // CHECK: br i1 %[[CMP]], label %[[BB1:.*]], label %[[BB2:.*]]
  %0 = llvm.mlir.addressof @x : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %cmp = llvm.icmp "eq" %1, %c1 : i32
  llvm.cond_br %cmp, ^bb1, ^bb2
// CHECK: [[BB1]]:
// CHECK: call void @variant1()
^bb1:
  llvm.call @variant1() : () -> ()
  llvm.br ^bb3
// CHECK: [[BB2]]:
// CHECK: call void @variant2()
^bb2:
  llvm.call @variant2() : () -> ()
  llvm.br ^bb3
^bb3:
  llvm.return
}

// CHECK-LABEL: define void @test_omp_dispatch()
llvm.func @test_omp_dispatch() {
  // CHECK: store i32 1, ptr @x
  %0 = llvm.mlir.addressof @x : !llvm.ptr
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.store %c1, %0 : i32, !llvm.ptr
  // CHECK: br label %omp.dispatch.region
  omp.dispatch {
    // CHECK: omp.dispatch.region:
    // CHECK-NEXT: call void @foo_dispatch()
    llvm.call @foo_dispatch() : () -> ()
    // CHECK-NEXT: br label %omp.region.cont
    omp.terminator
  }
  // CHECK: omp.region.cont:
  llvm.return
}

// CHECK-LABEL: define void @test_omp_dispatch_multiple()
llvm.func @test_omp_dispatch_multiple() {
  // CHECK: store i32 1, ptr @x
  %0 = llvm.mlir.addressof @x : !llvm.ptr
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.store %c1, %0 : i32, !llvm.ptr
  // CHECK: br label %omp.dispatch.region
  omp.dispatch {
    // CHECK: omp.dispatch.region:
    // CHECK-NEXT: call void @foo_dispatch()
    llvm.call @foo_dispatch() : () -> ()
    // CHECK-NEXT: br label %omp.region.cont
    omp.terminator
  }
  // CHECK: omp.region.cont:
  // CHECK: store i32 2, ptr @x
  %c2 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %c2, %0 : i32, !llvm.ptr
  // CHECK: br label %omp.dispatch.region2
  omp.dispatch {
    // CHECK: omp.dispatch.region2:
    // CHECK-NEXT: call void @foo_dispatch()
    llvm.call @foo_dispatch() : () -> ()
    // CHECK-NEXT: br label %omp.region.cont1
    omp.terminator
  }
  // CHECK: omp.region.cont1:
  llvm.return
}
