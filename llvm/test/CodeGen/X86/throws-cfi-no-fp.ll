; RUN: llc %s -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__clang_call_terminate = comdat any

@_ZL11ShouldThrow = internal unnamed_addr global i1 false, align 1
@_ZTIi = external constant ptr
@str = private unnamed_addr constant [20 x i8] c"Threw an exception!\00"

; Function Attrs: uwtable
define void @_Z6throwsv() personality ptr @__gxx_personality_v0 {

; CHECK-LABEL:   _Z6throwsv:
; CHECK:         popq	%rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    retq
; CHECK-NEXT:    .LBB0_1:
; CHECK-NEXT:    .cfi_def_cfa_offset 16

entry:
  %.b5 = load i1, ptr @_ZL11ShouldThrow, align 1
  br i1 %.b5, label %if.then, label %try.cont

if.then:                                          ; preds = %entry
  %exception = tail call ptr @__cxa_allocate_exception(i64 4)
  store i32 1, ptr %exception, align 16
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %if.then
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1)
  %puts = tail call i32 @puts(ptr @str)
  invoke void @__cxa_rethrow() #4
          to label %unreachable unwind label %lpad1

lpad1:                                            ; preds = %lpad
  %3 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

try.cont:                                         ; preds = %entry
  ret void

eh.resume:                                        ; preds = %lpad1
  resume { ptr, i32 } %3

terminate.lpad:                                   ; preds = %lpad1
  %4 = landingpad { ptr, i32 }
          catch ptr null
  %5 = extractvalue { ptr, i32 } %4, 0
  tail call void @__clang_call_terminate(ptr %5)
  unreachable

unreachable:                                      ; preds = %lpad, %if.then
  unreachable
}

declare ptr @__cxa_allocate_exception(i64)

declare void @__cxa_throw(ptr, ptr, ptr)

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_rethrow()

declare void @__cxa_end_catch()

; Function Attrs: noinline noreturn nounwind
declare void @__clang_call_terminate(ptr)

declare void @_ZSt9terminatev()


; Function Attrs: nounwind
declare i32 @puts(ptr nocapture readonly)

