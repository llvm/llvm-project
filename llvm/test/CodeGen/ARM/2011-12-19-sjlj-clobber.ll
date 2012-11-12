; RUN: llc < %s -O0 -mtriple=thumbv7-apple-ios | FileCheck %s

; Radar 10567930: Make sure that all the caller-saved registers are saved and
; restored in a function with setjmp/longjmp EH.  In particular, r6 was not
; being saved here.
; CHECK: push {r4, r5, r6, r7, lr}

%0 = type opaque
%struct.NSConstantString = type { i32*, i32, i8*, i32 }

define i32 @asdf(i32 %a, i32 %b, i8** %c, i8* %d) {
bb:
  %tmp = alloca i32, align 4
  %tmp1 = alloca i32, align 4
  %tmp2 = alloca i8*, align 4
  %tmp3 = alloca i1
  %myException = alloca %0*, align 4
  %tmp4 = alloca i8*
  %tmp5 = alloca i32
  %exception = alloca %0*, align 4
  store i32 %a, i32* %tmp, align 4
  store i32 %b, i32* %tmp1, align 4
  store i8* %d, i8** %tmp2, align 4
  store i1 false, i1* %tmp3
  %tmp7 = load i8** %c
  %tmp10 = invoke %0* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %0* (i8*, i8*, %0*)*)(i8* %tmp7, i8* %d, %0* null)
          to label %bb11 unwind label %bb15

bb11:                                             ; preds = %bb
  store %0* %tmp10, %0** %myException, align 4
  %tmp12 = load %0** %myException, align 4
  %tmp13 = bitcast %0* %tmp12 to i8*
  invoke void @objc_exception_throw(i8* %tmp13) noreturn
          to label %bb14 unwind label %bb15

bb14:                                             ; preds = %bb11
  unreachable

bb15:                                             ; preds = %bb11, %bb
  %tmp16 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          catch i8* null
  %tmp17 = extractvalue { i8*, i32 } %tmp16, 0
  store i8* %tmp17, i8** %tmp4
  %tmp18 = extractvalue { i8*, i32 } %tmp16, 1
  store i32 %tmp18, i32* %tmp5
  store i1 true, i1* %tmp3
  br label %bb56

bb56:
  unreachable
}

declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind
declare i32 @__objc_personality_v0(...)
declare void @objc_exception_throw(i8*)
