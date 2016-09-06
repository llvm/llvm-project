; RUN: opt -gvn-hoist -S < %s | FileCheck %s

; Check that convergent calls are not hoisted.
;
; CHECK-LABEL: @no_convergent_func_hoisting(
; CHECK: if.then:
; CHECK: call float @convergent_func(

; CHECK: if.else:
; CHECK: call float @convergent_func(
define float @no_convergent_func_hoisting(float %d, float %min, float %max, float %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %sub1 = fsub float %max, %a
  %mul2 = call float @convergent_func(float %sub1, float %div)
  br label %if.end

if.else:
  %sub5 = fsub float %max, %a
  %mul6 = call float @convergent_func(float %sub5, float %div)
  br label %if.end

if.end:
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %add = fadd float %tmax.0, 10.0
  ret float %add
}

; The call site is convergent but the declaration is not.
; CHECK-LABEL: @no_convergent_call_hoisting(

; CHECK: if.then:
; CHECK: call float @func(

; CHECK: if.else:
; CHECK: call float @func(
define float @no_convergent_call_hoisting(float %d, float %min, float %max, float %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %sub1 = fsub float %max, %a
  %mul2 = call float @func(float %sub1, float %div) #0
  br label %if.end

if.else:
  %sub5 = fsub float %max, %a
  %mul6 = call float @func(float %sub5, float %div) #0
  br label %if.end

if.end:
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %add = fadd float %tmax.0, 10.0
  ret float %add
}

; The call site is convergent but the declaration is not.
; CHECK-LABEL: @call_hoisting(
; CHECK: call float @func(
; CHECK-NOT: call float @func(
define float @call_hoisting(float %d, float %min, float %max, float %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %sub1 = fsub float %max, %a
  %mul2 = call float @func(float %sub1, float %div)
  br label %if.end

if.else:
  %sub5 = fsub float %max, %a
  %mul6 = call float @func(float %sub5, float %div)
  br label %if.end

if.end:
  %tmax.0 = phi float [ %mul2, %if.then ], [ %mul6, %if.else ]
  %add = fadd float %tmax.0, 10.0
  ret float %add
}

declare float @convergent_func(float, float) #0
declare float @func(float, float) #1

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind readnone }
