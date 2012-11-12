; RUN: llc < %s -mtriple=i386-apple-darwin10.0 -mcpu=core2  -relocation-model=dynamic-no-pic --asm-verbose=0   | FileCheck -check-prefix=I386 %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10.0 -mcpu=core2  -relocation-model=dynamic-no-pic --asm-verbose=0 | FileCheck -check-prefix=CORE2 %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10.0 -mcpu=corei7 -relocation-model=dynamic-no-pic --asm-verbose=0 | FileCheck -check-prefix=COREI7 %s

@.str1 = internal constant [31 x i8] c"DHRYSTONE PROGRAM, SOME STRING\00", align 8
@.str3 = internal constant [31 x i8] c"DHRYSTONE PROGRAM, 2'ND STRING\00", align 8

define void @func() nounwind ssp {
entry:
  %String2Loc = alloca [31 x i8], align 1
  br label %bb

bb:                                               ; preds = %bb, %entry
  %String2Loc9 = getelementptr inbounds [31 x i8]* %String2Loc, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %String2Loc9, i8* getelementptr inbounds ([31 x i8]* @.str3, i64 0, i64 0), i64 31, i32 1, i1 false)
  br label %bb

return:                                           ; No predecessors!
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; I386: calll {{_?}}memcpy

; CORE2: movabsq
; CORE2: movabsq
; CORE2: movabsq

; COREI7: movups _.str3

; CORE2: .section
; CORE2: .align  3
; CORE2-NEXT: _.str1:
; CORE2-NEXT: .asciz "DHRYSTONE PROGRAM, SOME STRING"
; CORE2: .align 3
; CORE2-NEXT: _.str3:
