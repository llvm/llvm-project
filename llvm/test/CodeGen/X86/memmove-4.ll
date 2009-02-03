; RUN: llvm-as < %s | llc | not grep call

target triple = "i686-pc-linux-gnu"

define void @a(i8* %a, i8* %b) nounwind {
        %tmp2 = bitcast i8* %a to i8*
        %tmp3 = bitcast i8* %b to i8*
        tail call void @llvm.memmove.i32( i8* %tmp2, i8* %tmp3, i32 12, i32 4 )
        ret void
}

declare void @llvm.memmove.i32(i8*, i8*, i32, i32)
