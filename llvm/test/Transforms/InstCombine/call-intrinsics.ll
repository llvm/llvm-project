; RUN: opt < %s -instcombine | llvm-dis

@X = global i8 0                ; <i8*> [#uses=3]
@Y = global i8 12               ; <i8*> [#uses=2]

declare void @llvm.memmove.i32(i8*, i8*, i32, i32)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @llvm.memset.i32(i8*, i8, i32, i32)

define void @zero_byte_test() {
        ; These process zero bytes, so they are a noop.
        call void @llvm.memmove.i32( i8* @X, i8* @Y, i32 0, i32 100 )
        call void @llvm.memcpy.i32( i8* @X, i8* @Y, i32 0, i32 100 )
        call void @llvm.memset.i32( i8* @X, i8 123, i32 0, i32 100 )
        ret void
}

