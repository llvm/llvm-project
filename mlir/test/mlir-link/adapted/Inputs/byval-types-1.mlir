module {
  llvm.func @baz(!llvm.ptr {llvm.byval = !llvm.struct<"struct", (i32, i8)>})
  llvm.func @foo(%arg0: !llvm.ptr {llvm.byval = !llvm.struct<"struct", (i32, i8)>}) {
    llvm.call @baz(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
}
