module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @c1 any
  }
  llvm.mlir.global weak hidden @v1(0 : i32) {addr_space = 0 : i32, dso_local} : i32
  llvm.mlir.global weak protected @v2(0 : i32) {addr_space = 0 : i32, dso_local} : i32
  llvm.mlir.global weak hidden @v3(0 : i32) {addr_space = 0 : i32, dso_local} : i32
  llvm.mlir.global external hidden @v4(1 : i32) comdat(@__llvm_global_comdat::@c1) {addr_space = 0 : i32, dso_local} : i32
  llvm.mlir.alias weak hidden @a1 {dso_local} : i32 {
    %0 = llvm.mlir.addressof @v1 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.alias weak protected @a2 {dso_local} : i32 {
    %0 = llvm.mlir.addressof @v2 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.alias weak hidden @a3 {dso_local} : i32 {
    %0 = llvm.mlir.addressof @v3 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func weak hidden @f1() attributes {dso_local} {
    llvm.return
  }
  llvm.func weak protected @f2() attributes {dso_local} {
    llvm.return
  }
  llvm.func weak hidden @f3() attributes {dso_local} {
    llvm.return
  }
}
