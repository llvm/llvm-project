// RUN: mlir-opt -pass-pipeline="builtin.module(llvm.func(fold-llvm-inttoptr-ptrtoint{address-space-bitwidths=64}))" %s | FileCheck %s --check-prefixes=CHECK-64BIT,CHECK-ALL
// RUN: mlir-opt -pass-pipeline="builtin.module(llvm.func(fold-llvm-inttoptr-ptrtoint{address-space-bitwidths=32}))" %s | FileCheck %s --check-prefixes=CHECK-32BIT,CHECK-ALL
// RUN: mlir-opt -pass-pipeline="builtin.module(llvm.func(fold-llvm-inttoptr-ptrtoint{address-space-bitwidths=64,32}))" %s | FileCheck %s --check-prefixes=CHECK-MULTI-ADDRSPACE,CHECK-ALL
// RUN: mlir-opt -pass-pipeline="builtin.module(llvm.func(fold-llvm-inttoptr-ptrtoint))" %s | FileCheck %s --check-prefixes=CHECK-DISABLED,CHECK-ALL


// CHECK-ALL-LABEL: @test_inttoptr_ptrtoint_fold_64bit
//  CHECK-ALL-SAME: (%[[ARG:.+]]: !llvm.ptr)
llvm.func @test_inttoptr_ptrtoint_fold_64bit(%arg0: !llvm.ptr) -> !llvm.ptr {
  // CHECK-64BIT-NOT: llvm.ptrtoint
  // CHECK-64BIT-NOT: llvm.inttoptr
  //     CHECK-64BIT: llvm.return %[[ARG]]

  // CHECK-32BIT-NOT: llvm.ptrtoint
  // CHECK-32BIT-NOT: llvm.inttoptr
  //     CHECK-32BIT: llvm.return %[[ARG]]

  // CHECK-MULTI-ADDRSPACE-NOT: llvm.ptrtoint
  // CHECK-MULTI-ADDRSPACE-NOT: llvm.inttoptr
  //     CHECK-MULTI-ADDRSPACE: llvm.return %[[ARG]]

  // CHECK-DISABLED: %[[INT:.+]] = llvm.ptrtoint %[[ARG]]
  // CHECK-DISABLED: %[[PTR:.+]] = llvm.inttoptr %[[INT]]
  // CHECK-DISABLED: llvm.return %[[PTR]]

  %0 = llvm.ptrtoint %arg0 : !llvm.ptr to i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// CHECK-ALL-LABEL: @test_ptrtoint_inttoptr_fold_64bit
//  CHECK-ALL-SAME: (%[[ARG:.+]]: i64)
llvm.func @test_ptrtoint_inttoptr_fold_64bit(%arg0: i64) -> i64 {
  // CHECK-64BIT-NOT: llvm.inttoptr
  // CHECK-64BIT-NOT: llvm.ptrtoint
  //     CHECK-64BIT: llvm.return %[[ARG]]

  // CHECK-32BIT: %[[INT:.+]] = llvm.inttoptr %[[ARG]]
  // CHECK-32BIT: %[[PTR:.+]] = llvm.ptrtoint %[[INT]]
  // CHECK-32BIT: llvm.return %[[PTR]]

  // CHECK-MULTI-ADDRSPACE-NOT: llvm.inttoptr
  // CHECK-MULTI-ADDRSPACE-NOT: llvm.ptrtoint
  //     CHECK-MULTI-ADDRSPACE: llvm.return %[[ARG]]

  // CHECK-DISABLED: %[[INT:.+]] = llvm.inttoptr %[[ARG]]
  // CHECK-DISABLED: %[[PTR:.+]] = llvm.ptrtoint %[[INT]]
  // CHECK-DISABLED: llvm.return %[[PTR]]

  %0 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
  %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
  llvm.return %1 : i64
}

// CHECK-ALL-LABEL: @test_inttoptr_ptrtoint_fold_addrspace1_32bit
//  CHECK-ALL-SAME: (%[[ARG:.+]]: !llvm.ptr<1>)
llvm.func @test_inttoptr_ptrtoint_fold_addrspace1_32bit(%arg0: !llvm.ptr<1>) -> !llvm.ptr<1> {
  // CHECK-64BIT: %[[INT:.+]] = llvm.ptrtoint %[[ARG]]
  // CHECK-64BIT: %[[PTR:.+]] = llvm.inttoptr %[[INT]]
  // CHECK-64BIT: llvm.return %[[PTR]]

  // CHECK-32BIT: %[[INT:.+]] = llvm.ptrtoint %[[ARG]]
  // CHECK-32BIT: %[[PTR:.+]] = llvm.inttoptr %[[INT]]
  // CHECK-32BIT: llvm.return %[[PTR]]

  // CHECK-MULTI-ADDRSPACE-NOT: llvm.ptrtoint
  // CHECK-MULTI-ADDRSPACE-NOT: llvm.inttoptr
  //     CHECK-MULTI-ADDRSPACE: llvm.return %[[ARG]]

  // CHECK-DISABLED: %[[INT:.+]] = llvm.ptrtoint %[[ARG]]
  // CHECK-DISABLED: %[[PTR:.+]] = llvm.inttoptr %[[INT]]
  // CHECK-DISABLED: llvm.return %[[PTR]]

  %0 = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i32
  %1 = llvm.inttoptr %0 : i32 to !llvm.ptr<1>
  llvm.return %1 : !llvm.ptr<1>
}

// CHECK-ALL-LABEL: @test_inttoptr_ptrtoint_type_mismatch
//  CHECK-ALL-SAME: (%[[ARG:.+]]: i64)
llvm.func @test_inttoptr_ptrtoint_type_mismatch(%arg0: i64) -> i32 {
  // CHECK-ALL: %[[INT:.+]] = llvm.inttoptr %[[ARG]]
  // CHECK-ALL: %[[PTR:.+]] = llvm.ptrtoint %[[INT]]
  // CHECK-ALL: llvm.return %[[PTR]]

  %0 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
  %1 = llvm.ptrtoint %0 : !llvm.ptr to i32
  llvm.return %1 : i32
}

// CHECK-ALL-LABEL: @test_ptrtoint_inttoptr_type_mismatch
//  CHECK-ALL-SAME: (%[[ARG:.+]]: !llvm.ptr<1>)
llvm.func @test_ptrtoint_inttoptr_type_mismatch(%arg0: !llvm.ptr<1>) -> !llvm.ptr<0> {
  // CHECK-ALL: %[[INT:.+]] = llvm.ptrtoint %[[ARG]]
  // CHECK-ALL: %[[PTR:.+]] = llvm.inttoptr %[[INT]]
  // CHECK-ALL: llvm.return %[[PTR]]
  %0 = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr<0>
  llvm.return %1 : !llvm.ptr<0>
}
