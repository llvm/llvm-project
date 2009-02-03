; RUN: llvm-as < %s | opt -anders-aa -gvn | llvm-dis | not grep undef
; PR2169

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind
declare void @use(i8)

define void @f(i8* %x) {
entry:
        %copy = alloca i8               ; <i8*> [#uses=6]
        call void @llvm.memcpy.i32( i8* %copy, i8* %x, i32 1, i32 4 )
        %tmp = load i8* %copy           ; <i8> [#uses=1]
        call void @use(i8 %tmp)
        ret void
}
