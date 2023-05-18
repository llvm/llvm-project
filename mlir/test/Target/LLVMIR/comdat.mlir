// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.comdat @__llvm_comdat {
  // CHECK-DAG: $[[ANY:.*]] = comdat any
  llvm.comdat_selector @any any
  // CHECK-DAG: $[[EXACT:.*]] = comdat exactmatch
  llvm.comdat_selector @exactmatch exactmatch
  // CHECK-DAG: $[[LARGEST:.*]] = comdat largest
  llvm.comdat_selector @largest largest
  // CHECK-DAG: $[[NODEDUP:.*]] = comdat nodeduplicate
  llvm.comdat_selector @nodeduplicate nodeduplicate
  // CHECK-DAG: $[[SAME:.*]] = comdat samesize
  llvm.comdat_selector @samesize samesize
}

// CHECK: @any = internal constant i64 1, comdat
llvm.mlir.global internal constant @any(1 : i64) comdat(@__llvm_comdat::@any) : i64
// CHECK: @any_global = internal constant i64 1, comdat($[[ANY]])
llvm.mlir.global internal constant @any_global(1 : i64) comdat(@__llvm_comdat::@any) : i64
// CHECK: @exact_global = internal constant i64 1, comdat($[[EXACT]])
llvm.mlir.global internal constant @exact_global(1 : i64) comdat(@__llvm_comdat::@exactmatch) : i64
// CHECK: @largest_global = internal constant i64 1, comdat($[[LARGEST]])
llvm.mlir.global internal constant @largest_global(1 : i64) comdat(@__llvm_comdat::@largest) : i64

// CHECK: define void @nodeduplicate() comdat
llvm.func @nodeduplicate() comdat(@__llvm_comdat::@nodeduplicate) { llvm.return }
// CHECK: define void @nodeduplicate_func() comdat($[[NODEDUP]])
llvm.func @nodeduplicate_func() comdat(@__llvm_comdat::@nodeduplicate) { llvm.return }
// CHECK: define void @samesize_func() comdat($[[SAME]])
llvm.func @samesize_func() comdat(@__llvm_comdat::@samesize) { llvm.return }
