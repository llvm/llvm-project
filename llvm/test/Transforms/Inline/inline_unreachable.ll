; RUN: opt < %s -inline -S | FileCheck %s

@a = global i32 4
@_ZTIi = external global i8*

; CHECK-LABEL: callSimpleFunction
; CHECK: call i32 @simpleFunction
define i32 @callSimpleFunction(i32 %idx, i32 %limit) {
entry:
  %cmp = icmp sge i32 %idx, %limit
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %s = call i32 @simpleFunction(i32 %idx)
  store i32 %s, i32* @a
  unreachable

if.end:
  ret i32 %idx
}

; CHECK-LABEL: callSmallFunction
; CHECK-NOT: call i32 @smallFunction
define i32 @callSmallFunction(i32 %idx, i32 %limit) {
entry:
  %cmp = icmp sge i32 %idx, %limit
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %s = call i32 @smallFunction(i32 %idx)
  store i32 %s, i32* @a
  unreachable

if.end:
  ret i32 %idx
}

; CHECK-LABEL: throwSimpleException
; CHECK: invoke i32 @simpleFunction
define i32 @throwSimpleException(i32 %idx, i32 %limit) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %cmp = icmp sge i32 %idx, %limit
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %exception = call i8* @__cxa_allocate_exception(i64 1) #0
  invoke i32 @simpleFunction(i32 %idx)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi  to i8*), i8* null) #1
  unreachable

lpad:                                             ; preds = %if.then
  %ll = landingpad { i8*, i32 }
          cleanup
  ret i32 %idx

if.end:                                           ; preds = %entry
  ret i32 %idx
}

; CHECK-LABEL: throwSmallException
; CHECK-NOT: invoke i32 @smallFunction
define i32 @throwSmallException(i32 %idx, i32 %limit) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %cmp = icmp sge i32 %idx, %limit
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %exception = call i8* @__cxa_allocate_exception(i64 1) #0
  invoke i32 @smallFunction(i32 %idx)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi  to i8*), i8* null) #1
  unreachable

lpad:                                             ; preds = %if.then
  %ll = landingpad { i8*, i32 }
          cleanup
  ret i32 %idx

if.end:                                           ; preds = %entry
  ret i32 %idx
}

define i32 @simpleFunction(i32 %a) #0 {
entry:
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32, i32* @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32, i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32, i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32, i32* @a
  %x5 = add i32 %x4, %a5
  %a6 = load volatile i32, i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32, i32* @a
  %x7 = add i32 %x6, %a6
  %a8 = load volatile i32, i32* @a
  %x8 = add i32 %x7, %a8
  %a9 = load volatile i32, i32* @a
  %x9 = add i32 %x8, %a9
  %a10 = load volatile i32, i32* @a
  %x10 = add i32 %x9, %a10
  %a11 = load volatile i32, i32* @a
  %x11 = add i32 %x10, %a11
  %a12 = load volatile i32, i32* @a
  %x12 = add i32 %x11, %a12
  %add = add i32 %x12, %a
  ret i32 %add
}

define i32 @smallFunction(i32 %a) {
entry:
  %r = load volatile i32, i32* @a
  ret i32 %r
}

attributes #0 = { nounwind }
attributes #1 = { noreturn }

declare i8* @__cxa_allocate_exception(i64)
declare i32 @__gxx_personality_v0(...)
declare void @__cxa_throw(i8*, i8*, i8*)

