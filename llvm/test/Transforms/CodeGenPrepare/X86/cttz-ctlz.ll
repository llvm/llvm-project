; RUN: opt -S -codegenprepare < %s | FileCheck %s --check-prefix=SLOW
; RUN: opt -S -codegenprepare -mattr=+bmi < %s | FileCheck %s --check-prefix=FAST_TZ
; RUN: opt -S -codegenprepare -mattr=+lzcnt < %s | FileCheck %s --check-prefix=FAST_LZ

target triple = "x86_64-unknown-unknown"
target datalayout = "e-n32:64"

; If the intrinsic is cheap, nothing should change.
; If the intrinsic is expensive, check if the input is zero to avoid the call. 
; This is undoing speculation that may have been created by SimplifyCFG + InstCombine.

define i64 @cttz(i64 %A) {
entry:
  %z = call i64 @llvm.cttz.i64(i64 %A, i1 false)
  ret i64 %z

; SLOW-LABEL: @cttz(
; SLOW: entry:
; SLOW:   %cmpz = icmp eq i64 %A, 0
; SLOW:   br i1 %cmpz, label %cond.end, label %cond.false
; SLOW: cond.false:
; SLOW:   %z = call i64 @llvm.cttz.i64(i64 %A, i1 true)
; SLOW:   br label %cond.end
; SLOW: cond.end:
; SLOW:   %ctz = phi i64 [ 64, %entry ], [ %z, %cond.false ]
; SLOW:   ret i64 %ctz

; FAST_TZ-LABEL: @cttz(
; FAST_TZ:  %z = call i64 @llvm.cttz.i64(i64 %A, i1 false)
; FAST_TZ:  ret i64 %z
}

define i64 @ctlz(i64 %A) {
entry:
  %z = call i64 @llvm.ctlz.i64(i64 %A, i1 false)
  ret i64 %z

; SLOW-LABEL: @ctlz(
; SLOW: entry:
; SLOW:   %cmpz = icmp eq i64 %A, 0
; SLOW:   br i1 %cmpz, label %cond.end, label %cond.false
; SLOW: cond.false:
; SLOW:   %z = call i64 @llvm.ctlz.i64(i64 %A, i1 true)
; SLOW:   br label %cond.end
; SLOW: cond.end:
; SLOW:   %ctz = phi i64 [ 64, %entry ], [ %z, %cond.false ]
; SLOW:   ret i64 %ctz

; FAST_LZ-LABEL: @ctlz(
; FAST_LZ:  %z = call i64 @llvm.ctlz.i64(i64 %A, i1 false)
; FAST_LZ:  ret i64 %z
}

declare i64 @llvm.cttz.i64(i64, i1)
declare i64 @llvm.ctlz.i64(i64, i1)

