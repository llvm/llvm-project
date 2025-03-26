// RUN: mlir-link %s %S/Inputs/testlink.mlir -o - | FileCheck %s

// type renaming is not yet supported
// XFAIL: *

module {
  llvm.mlir.global external @S1GV() {addr_space = 0 : i32} : !llvm.struct<"Struct1", opaque>
  // CHECK: !llvm.struct<"Ty1.1", (ptr)>
  llvm.mlir.global external @GVTy1() {addr_space = 0 : i32} : !llvm.struct<"Ty1", opaque>
  // CHECK: !llvm.struct<"Ty2", (ptr, ptr)>
  llvm.mlir.global external @GVTy2() {addr_space = 0 : i32} : !llvm.struct<"Ty2", (ptr, ptr)> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.struct<"Ty2", (ptr, ptr)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<"Ty2", (ptr, ptr)>
    %3 = llvm.insertvalue %0, %2[1] : !llvm.struct<"Ty2", (ptr, ptr)>
    llvm.return %3 : !llvm.struct<"Ty2", (ptr, ptr)>
  }
  // CHECK-DAG: llvm.mlir.global external @MyIntList {{.*}} !llvm.struct<"intlist", (ptr, i32)> {
  llvm.mlir.global external @MyIntList() {addr_space = 0 : i32} : !llvm.struct<"intlist", (ptr, i32)> {
    %0 = llvm.mlir.constant(17 : i32) : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.struct<"intlist", (ptr, i32)>
    %3 = llvm.insertvalue %1, %2[0] : !llvm.struct<"intlist", (ptr, i32)>
    %4 = llvm.insertvalue %0, %3[1] : !llvm.struct<"intlist", (ptr, i32)>
    llvm.return %4 : !llvm.struct<"intlist", (ptr, i32)>
  }

  // CHECK-DAG: llvm.mlir.global external @mlir.llvm.nameless_global_0() {addr_space = 0 : i32} : i32
  llvm.mlir.global external @mlir.llvm.nameless_global_0() {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global external @Inte(1 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @Inte(1 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global internal constant @Intern1(1 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global internal constant @Intern1(42 : i32) {addr_space = 0 : i32, dso_local} : i32
  llvm.mlir.global external @UseIntern1() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @Intern1 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  // CHECK-DAG: llvm.mlir.global internal constant @Intern2.{{[0-9]+}}(1 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global internal constant @Intern2(792 : i32) {addr_space = 0 : i32, dso_local} : i32
  llvm.mlir.global external @UseIntern2() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @Intern2 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  // CHECK-DAG: llvm.mlir.global linkonce @MyVarPtr {{.*}} {
  llvm.mlir.global linkonce @MyVarPtr() {addr_space = 0 : i32} : !llvm.struct<(ptr)> {
    %0 = llvm.mlir.addressof @MyVar : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.struct<(ptr)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr)>
    llvm.return %2 : !llvm.struct<(ptr)>
  }
  llvm.mlir.global external @UseMyVarPtr() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @MyVarPtr : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  // CHECK-DAG: llvm.mlir.global linkonce @MyVar(4 : i32)
  llvm.mlir.global external @MyVar() {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global linkonce @AConst(1234 : i32)
  llvm.mlir.global linkonce constant @AConst(123 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global internal constant @Intern1.{{[0-9]+}}(52 : i32)
  // CHECK-DAG: llvm.mlir.global external constant @Intern2(12345 : i32)
  // CHECK-DAG: llvm.mlir.global external constant @MyIntListPtr
  // CHECK-DAG: llvm.mlir.global external constant @mlir.llvm.nameless_global_0(412 : i32)

  llvm.func @use0() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @foo(i32) -> i32
  llvm.func @print(i32)
  llvm.func @main() {
    %0 = llvm.mlir.addressof @MyVar : !llvm.ptr
    %1 = llvm.mlir.addressof @MyIntList : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(5 : i32) : i32
    %5 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.call @print(%5) : (i32) -> ()
    %6 = llvm.getelementptr %1[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"intlist", (ptr, i32)>
    %7 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.call @print(%7) : (i32) -> ()
    %8 = llvm.call @foo(%4) : (i32) -> i32
    %9 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.call @print(%9) : (i32) -> ()
    %10 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.call @print(%10) : (i32) -> ()
    llvm.return
  }
  llvm.func internal @testintern() attributes {dso_local} {
    llvm.return
  }
  llvm.func internal @Testintern() attributes {dso_local} {
    llvm.return
  }
  llvm.func @testIntern() {
    llvm.return
  }
  llvm.func @VecSizeCrash(%arg0: !llvm.struct<"VecSize", (vector<5xi32>)>) {
    llvm.return
  }
}
