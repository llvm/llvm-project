; RUN: llc < %s | FileCheck %s
; RUN: llc -O0 < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @f(i32)
declare i32 @__C_specific_handler(...)
declare i32 @llvm.eh.exceptioncode(token)

define void @ehcode() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @f(i32 0)
          to label %__try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %pad = catchpad [i8* null]
          to label %__except unwind label %catchendblock

__except:                                         ; preds = %catch.dispatch
  catchret %pad to label %__except.1

__except.1:                                       ; preds = %__except
  %code = call i32 @llvm.eh.exceptioncode(token %pad)
  call void @f(i32 %code)
  br label %__try.cont

__try.cont:                                       ; preds = %entry, %__except.1
  ret void

catchendblock:                                    ; preds = %catch.dispatch
  catchendpad unwind to caller
}

; CHECK-LABEL: ehcode:
; CHECK: xorl %ecx, %ecx
; CHECK: callq f

; CHECK: # %__except
; CHECK: movl %eax, %ecx
; CHECK-NEXT: callq f
