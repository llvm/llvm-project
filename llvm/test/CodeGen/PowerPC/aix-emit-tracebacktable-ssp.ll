; RUN: llc -aix-ssp-tb-bit -mtriple=powerpc64-ibm-aix-xcoff -O0 < %s | FileCheck %s

; CHECK-LABEL: f:
; CHECK: __ssp_canary_word
; CHECK: TB_SSP_CANARY
define i32 @f() #0 personality ptr @__xlcxx_personality_v1 {
  invoke i32 undef(ptr undef)
    to label %invoke unwind label %lpad

  invoke:
    %var = alloca i32, align 4
    store i32 0, ptr %var, align 4
    %gep = getelementptr inbounds i32, ptr %var, i32 1
    %ret = load i32, ptr %gep, align 4
    ret i32 %ret
  lpad:
    landingpad { ptr, i32 }
  catch ptr null
    unreachable

}

; CHECK-LABEL: f2:
; CHECK: __ssp_canary_word
; Not emitting traceback bit when no unwinding needed.
; CHECK-NOT: TB_SSP_CANARY
define i32 @f2() #0 {
  %var = alloca i32, align 4
  store i32 0, ptr %var, align 4
  %gep = getelementptr inbounds i32, ptr %var, i32 1
  %ret = load i32, ptr %gep, align 4
  ret i32 %ret
}

declare i32 @__xlcxx_personality_v1(...)
attributes #0 = { sspstrong }

