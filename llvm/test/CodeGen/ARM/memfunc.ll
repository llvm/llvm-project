; RUN: llvm-as < %s | llc -march=arm

define void @f() {
entry:
        call void @llvm.memmove.i32( i8* null, i8* null, i32 64, i32 0 )
        call void @llvm.memcpy.i32( i8* null, i8* null, i32 64, i32 0 )
        call void @llvm.memset.i32( i8* null, i8 64, i32 0, i32 0 )
        unreachable
}

declare void @llvm.memmove.i32(i8*, i8*, i32, i32)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @llvm.memset.i32(i8*, i8, i32, i32)

