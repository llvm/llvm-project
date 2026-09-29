; RUN: llc < %s | FileCheck %s
;
; PR27612. The following spill is hoisted from two locations: the fall
; through succ block and the landingpad block of a call which may throw
; exception. If it is not hoisted before the call, the spill will be
; missing on the landingpad path.
;
; CHECK-LABEL: _Z3foov:
; CHECK: movq  %rbx, (%rsp)          # 8-byte Spill
; CHECK-NEXT: callq _Z3goov

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = dso_local global [20 x i64] zeroinitializer, align 16
@_ZTIi = external constant ptr

; Function Attrs: uwtable
define dso_local void @_Z3foov() personality ptr @__gxx_personality_v0 {
entry:
  %tmp = load i64, ptr getelementptr inbounds ([20 x i64], ptr @a, i64 0, i64 1), align 8
  invoke void @_Z3goov()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %tmp1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  %tmp2 = extractvalue { ptr, i32 } %tmp1, 1
  %tmp3 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %tmp2, %tmp3
  br i1 %matches, label %catch, label %ehcleanup

catch:                                            ; preds = %lpad
  %tmp4 = extractvalue { ptr, i32 } %tmp1, 0
  %tmp5 = tail call ptr @__cxa_begin_catch(ptr %tmp4)
  store i64 %tmp, ptr getelementptr inbounds ([20 x i64], ptr @a, i64 0, i64 2), align 16
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{memory},~{dirflag},~{fpsr},~{flags}"()
  store i64 %tmp, ptr getelementptr inbounds ([20 x i64], ptr @a, i64 0, i64 3), align 8
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %catch, %entry
  store i64 %tmp, ptr getelementptr inbounds ([20 x i64], ptr @a, i64 0, i64 4), align 16
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{memory},~{dirflag},~{fpsr},~{flags}"()
  store i64 %tmp, ptr getelementptr inbounds ([20 x i64], ptr @a, i64 0, i64 5), align 8
  ret void

ehcleanup:                                        ; preds = %lpad
  resume { ptr, i32 } %tmp1
}

declare void @_Z3goov()

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()
