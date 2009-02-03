; RUN: llvm-as %s -o /dev/null -f

define void @test(i32* %P, i32* %Q) {
entry:
        %tmp.1 = bitcast i32* %P to i8*         ; <i8*> [#uses=3]
        %tmp.3 = bitcast i32* %Q to i8*         ; <i8*> [#uses=4]
        tail call void @llvm.memcpy.i32( i8* %tmp.1, i8* %tmp.3, i32 100000, i32 1 )
        tail call void @llvm.memcpy.i64( i8* %tmp.1, i8* %tmp.3, i64 100000, i32 1 )
        tail call void @llvm.memset.i32( i8* %tmp.3, i8 14, i32 10000, i32 0 )
        tail call void @llvm.memmove.i32( i8* %tmp.1, i8* %tmp.3, i32 123124, i32 1 )
        ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32)

declare void @llvm.memset.i32(i8*, i8, i32, i32)

declare void @llvm.memmove.i32(i8*, i8*, i32, i32)

