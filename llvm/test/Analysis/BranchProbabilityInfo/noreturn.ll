; Test the static branch probability heuristics for no-return functions.
; RUN: opt < %s -analyze -branch-prob | FileCheck %s
; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

declare void @abort() noreturn

define i32 @test1(i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test1'
entry:
  %cond = icmp eq i32 %a, 42
  br i1 %cond, label %exit, label %abort
; CHECK: edge entry -> exit probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK: edge entry -> abort probability is 0x00000800 / 0x80000000 = 0.00%

abort:
  call void @abort() noreturn
  unreachable

exit:
  ret i32 %b
}

define i32 @test2(i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test2'
entry:
  switch i32 %a, label %exit [i32 1, label %case_a
                              i32 2, label %case_b
                              i32 3, label %case_c
                              i32 4, label %case_d]
; CHECK: edge entry -> exit probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK: edge entry -> case_a probability is 0x00000200 / 0x80000000 = 0.00%
; CHECK: edge entry -> case_b probability is 0x00000200 / 0x80000000 = 0.00%
; CHECK: edge entry -> case_c probability is 0x00000200 / 0x80000000 = 0.00%
; CHECK: edge entry -> case_d probability is 0x00000200 / 0x80000000 = 0.00%

case_a:
  br label %case_b

case_b:
  br label %case_c

case_c:
  br label %case_d

case_d:
  call void @abort() noreturn
  unreachable

exit:
  ret i32 %b
}

define i32 @test3(i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test3'
; Make sure we unify across multiple conditional branches.
entry:
  %cond1 = icmp eq i32 %a, 42
  br i1 %cond1, label %exit, label %dom
; CHECK: edge entry -> exit probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK: edge entry -> dom probability is 0x00000800 / 0x80000000 = 0.00%

dom:
  %cond2 = icmp ult i32 %a, 42
  br i1 %cond2, label %idom1, label %idom2
; CHECK: edge dom -> idom1 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge dom -> idom2 probability is 0x40000000 / 0x80000000 = 50.00%

idom1:
  br label %abort

idom2:
  br label %abort

abort:
  call void @abort() noreturn
  unreachable

exit:
  ret i32 %b
}

@_ZTIi = external global i8*

; CHECK-LABEL: throwSmallException
; CHECK-NOT: invoke i32 @smallFunction
define i32 @throwSmallException(i32 %idx, i32 %limit) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %cmp = icmp sge i32 %idx, %limit
  br i1 %cmp, label %if.then, label %if.end
; CHECK: edge entry -> if.then probability is 0x00000800 / 0x80000000 = 0.00%
; CHECK: edge entry -> if.end probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]

if.then:                                          ; preds = %entry
  %exception = call i8* @__cxa_allocate_exception(i64 1) #0
  invoke i32 @smallFunction(i32 %idx)
          to label %invoke.cont unwind label %lpad
; CHECK: edge if.then -> invoke.cont probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK: edge if.then -> lpad probability is 0x00000800 / 0x80000000 = 0.00%

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

@a = global i32 4
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
