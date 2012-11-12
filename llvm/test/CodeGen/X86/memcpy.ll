; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core2 | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=DARWIN

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind


; Variable memcpy's should lower to calls.
define i8* @test1(i8* %a, i8* %b, i64 %n) nounwind {
entry:
	tail call void @llvm.memcpy.p0i8.p0i8.i64( i8* %a, i8* %b, i64 %n, i32 1, i1 0 )
	ret i8* %a
        
; LINUX: test1:
; LINUX: memcpy
}

; Variable memcpy's should lower to calls.
define i8* @test2(i64* %a, i64* %b, i64 %n) nounwind {
entry:
	%tmp14 = bitcast i64* %a to i8*
	%tmp25 = bitcast i64* %b to i8*
	tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp14, i8* %tmp25, i64 %n, i32 8, i1 0 )
	ret i8* %tmp14
        
; LINUX: test2:
; LINUX: memcpy
}

; Large constant memcpy's should lower to a call when optimizing for size.
; PR6623

; On the other hand, Darwin's definition of -Os is optimizing for size without
; hurting performance so it should just ignore optsize when expanding memcpy.
; rdar://8821501
define void @test3(i8* nocapture %A, i8* nocapture %B) nounwind optsize noredzone {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; LINUX: test3:
; LINUX: memcpy

; DARWIN: test3:
; DARWIN-NOT: memcpy
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
; DARWIN: movq
}

; Large constant memcpy's should be inlined when not optimizing for size.
define void @test4(i8* nocapture %A, i8* nocapture %B) nounwind noredzone {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; LINUX: test4:
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
; LINUX: movq
}


@.str = private unnamed_addr constant [30 x i8] c"\00aaaaaaaaaaaaaaaaaaaaaaaaaaaa\00", align 1

define void @test5(i8* nocapture %C) nounwind uwtable ssp {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([30 x i8]* @.str, i64 0, i64 0), i64 16, i32 1, i1 false)
  ret void

; DARWIN: movabsq	$7016996765293437281
; DARWIN: movabsq	$7016996765293437184
}


