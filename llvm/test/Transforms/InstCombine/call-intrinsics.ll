; RUN: opt < %s -instcombine | llvm-dis

@X = global i8 0                ; <i8*> [#uses=3]
@Y = global i8 12               ; <i8*> [#uses=2]

declare void @llvm.memmove.p0i8.p0i8.i32(i8*, i8*, i32, i1)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)

define void @zero_byte_test() {
        ; These process zero bytes, so they are a noop.
        call void @llvm.memmove.p0i8.p0i8.i32(i8* align 128 @X, i8* align 128 @Y, i32 0, i1 false )
        call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 128 @X, i8* align 128 @Y, i32 0, i1 false )
        call void @llvm.memset.p0i8.i32(i8* align 128 @X, i8 123, i32 0, i1 false )
        ret void
}

