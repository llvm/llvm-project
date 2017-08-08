; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...)

; On X86 1 is true and 0 is false, so we can't perform the combine:
; (and (setgt X,  true), (setgt Y,  true)) -> (setgt (or X, Y), true)
; This combine only works if the true value is -1.


;CHECK: cmpl
;CHECK: setl
;CHECK: cmpl
;CHECK: setl
;CHECK: orb
;CHECK: je

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
; Function Attrs: optsize ssp uwtable
define i32 @foo(i32 %a, i32 %b, i32 * %c) {
if.else429:
  %cmp.i1144 = icmp eq i32* %c, null
  %cmp430 = icmp slt i32 %a, 2
  %cmp432 = icmp slt i32 %b, 2
  %or.cond710 = or i1 %cmp430, %cmp432
  %or.cond710.not = xor i1 %or.cond710, true
  %brmerge1448 = or i1 %cmp.i1144, %or.cond710.not
  br i1 %brmerge1448, label %ret1, label %ret2

ret1:
  ret i32 0

ret2:
  ret i32 1
}

define i32 @main(i32 %argc, i8** nocapture readnone %argv) {
  %res = alloca i32, align 4
  %t = call i32 @foo(i32 1, i32 2, i32* %res) #3
  %v = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %t)
  ret i32 0
}



