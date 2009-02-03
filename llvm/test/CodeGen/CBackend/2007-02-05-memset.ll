; RUN: llvm-as < %s | llc -march=c
; PR1181
target datalayout = "e-p:64:64"
target triple = "x86_64-apple-darwin8"


declare void @llvm.memset.i64(i8*, i8, i64, i32)

define fastcc void @InitUser_data_unregistered() {
entry:
        tail call void @llvm.memset.i64( i8* null, i8 0, i64 65496, i32 1 )
        ret void
}
