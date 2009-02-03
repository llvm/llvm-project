; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin | grep stosl
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | grep movq | count 10

define void @bork() nounwind {
entry:
        call void @llvm.memset.i64( i8* null, i8 0, i64 80, i32 4 )
        ret void
}

declare void @llvm.memset.i64(i8*, i8, i64, i32) nounwind

