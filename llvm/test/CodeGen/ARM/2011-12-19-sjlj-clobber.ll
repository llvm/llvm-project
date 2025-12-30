; RUN: llc < %s -O0 -mtriple=thumbv7-apple-ios | FileCheck %s

; Radar 10567930: Make sure that all the caller-saved registers are saved and
; restored in a function with setjmp/longjmp EH.  In particular, r6 was not
; being saved here.
; CHECK: push.w {r4, r5, r6, r7, r8, r10, r11, lr}

%0 = type opaque
%struct.NSConstantString = type { ptr, i32, ptr, i32 }

define i32 @asdf(i32 %a, i32 %b, ptr %c, ptr %d) personality ptr @__objc_personality_v0 {
bb:
  %tmp = alloca i32, align 4
  %tmp1 = alloca i32, align 4
  %tmp2 = alloca ptr, align 4
  %tmp3 = alloca i1
  %myException = alloca ptr, align 4
  %tmp4 = alloca ptr
  %tmp5 = alloca i32
  %exception = alloca ptr, align 4
  store i32 %a, ptr %tmp, align 4
  store i32 %b, ptr %tmp1, align 4
  store ptr %d, ptr %tmp2, align 4
  store i1 false, ptr %tmp3
  %tmp7 = load ptr, ptr %c
  %tmp10 = invoke ptr @objc_msgSend(ptr %tmp7, ptr %d, ptr null)
          to label %bb11 unwind label %bb15

bb11:                                             ; preds = %bb
  store ptr %tmp10, ptr %myException, align 4
  %tmp12 = load ptr, ptr %myException, align 4
  invoke void @objc_exception_throw(ptr %tmp12) noreturn
          to label %bb14 unwind label %bb15

bb14:                                             ; preds = %bb11
  unreachable

bb15:                                             ; preds = %bb11, %bb
  %tmp16 = landingpad { ptr, i32 }
          catch ptr null
  %tmp17 = extractvalue { ptr, i32 } %tmp16, 0
  store ptr %tmp17, ptr %tmp4
  %tmp18 = extractvalue { ptr, i32 } %tmp16, 1
  store i32 %tmp18, ptr %tmp5
  store i1 true, ptr %tmp3
  br label %bb56

bb56:
  unreachable
}

declare ptr @objc_msgSend(ptr, ptr, ...) nonlazybind
declare i32 @__objc_personality_v0(...)
declare void @objc_exception_throw(ptr)
