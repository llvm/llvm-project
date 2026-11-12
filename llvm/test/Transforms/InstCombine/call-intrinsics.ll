; RUN: opt < %s -passes=instcombine | llvm-dis

@X = global i8 0
@Y = global i8 12

declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)

declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)

define void @zero_byte_test() {
  ; These process zero bytes, so they are a noop.
  call void @llvm.memmove.p0.p0.i32(ptr @X, ptr @Y, i32 0, i1 false )
  call void @llvm.memcpy.p0.p0.i32(ptr @X, ptr @Y, i32 0, i1 false )
  call void @llvm.memset.p0.i32(ptr @X, i8 123, i32 0, i1 false )
  ret void
}

