module {
  llvm.func @g(%arg0: !llvm.ptr {llvm.byref = !llvm.struct<"a", (i64)>}) {
    llvm.return
  }
  llvm.func @baz(!llvm.ptr {llvm.byref = !llvm.struct<"struct", (i32, i8)>})
  llvm.func @foo(%arg0: !llvm.ptr {llvm.byref = !llvm.struct<"struct", (i32, i8)>}) {
    llvm.call @baz(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
}
