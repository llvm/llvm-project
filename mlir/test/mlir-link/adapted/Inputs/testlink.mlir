module {
  llvm.mlir.global external @GVTy1() {addr_space = 0 : i32} : !llvm.struct<"Ty1", (ptr)> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.struct<"Ty1", (ptr)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<"Ty1", (ptr)>
    llvm.return %2 : !llvm.struct<"Ty1", (ptr)>
  }
  llvm.mlir.global external @GVTy2() {addr_space = 0 : i32} : !llvm.struct<"Ty2", opaque>
  llvm.mlir.global external @MyVar(4 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @MyIntList() {addr_space = 0 : i32} : !llvm.struct<"intlist", (ptr, i32)>
  llvm.mlir.global external constant @AConst(1234 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global internal constant @Intern1(52 : i32) {addr_space = 0 : i32, dso_local} : i32
  llvm.mlir.global external @Use2Intern1() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @Intern1 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global external constant @Intern2(12345 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external constant @MyIntListPtr() {addr_space = 0 : i32} : !llvm.struct<(ptr)> {
    %0 = llvm.mlir.addressof @MyIntList : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.struct<(ptr)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr)>
    llvm.return %2 : !llvm.struct<(ptr)>
  }
  llvm.mlir.global linkonce @MyVarPtr() {addr_space = 0 : i32} : !llvm.struct<(ptr)> {
    %0 = llvm.mlir.addressof @MyVar : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.struct<(ptr)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr)>
    llvm.return %2 : !llvm.struct<(ptr)>
  }
  llvm.mlir.global external constant @mlir.llvm.nameless_global_0(412 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @S1GV() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @foo(%arg0: i32) -> i32 {
    %0 = llvm.mlir.addressof @MyVar : !llvm.ptr
    %1 = llvm.mlir.addressof @MyIntList : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(12 : i32) : i32
    %5 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    llvm.store %arg0, %0 {alignment = 4 : i64} : i32, !llvm.ptr
    %6 = llvm.getelementptr %1[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"intlist", (ptr, i32)>
    llvm.store %4, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    %7 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %8 = llvm.add %7, %arg0 : i32
    llvm.return %8 : i32
  }
  llvm.func @unimp(f32, f64)
  llvm.func internal @testintern() attributes {dso_local} {
    llvm.return
  }
  llvm.func @Testintern() {
    llvm.return
  }
  llvm.func internal @testIntern() attributes {dso_local} {
    llvm.return
  }
  llvm.func @VecSizeCrash1(%arg0: !llvm.struct<"VecSize", (vector<10xi32>)>) {
    llvm.return
  }
}
