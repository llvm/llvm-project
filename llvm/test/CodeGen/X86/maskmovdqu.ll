; RUN: llc < %s -mtriple=i686--    -mattr=+sse2,-avx | grep -i edi
; RUN: llc < %s -mtriple=x86_64-- -mattr=+sse2,-avx | grep -i rdi
; RUN: llc < %s -mtriple=i686--    -mattr=+avx | grep -i edi
; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx | grep -i rdi
; rdar://6573467

define void @test(<16 x i8> %a, <16 x i8> %b, i32 %dummy, i8* %c) nounwind {
entry:
	tail call void @llvm.x86.sse2.maskmov.dqu( <16 x i8> %a, <16 x i8> %b, i8* %c )
	ret void
}

declare void @llvm.x86.sse2.maskmov.dqu(<16 x i8>, <16 x i8>, i8*) nounwind
