; RUN: llvm-as < %s | llc | FileCheck %s
; PR 5247
; check that cmp/test is not scheduled before the add
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@.str76843 = external dso_local constant [45 x i8]          ; <ptr> [#uses=1]
@__profiling_callsite_timestamps_live = external dso_local global [1216 x i64] ; <ptr> [#uses=2]

define i32 @cl_init(i32 %initoptions) nounwind {
entry:
  %retval.i = alloca i32                          ; <ptr> [#uses=3]
  %retval = alloca i32                            ; <ptr> [#uses=2]
  %initoptions.addr = alloca i32                  ; <ptr> [#uses=2]
  tail call void asm sideeffect "cpuid", "~{ax},~{bx},~{cx},~{dx},~{memory},~{dirflag},~{fpsr},~{flags}"() nounwind
  %0 = tail call i64 @llvm.readcyclecounter() nounwind ; <i64> [#uses=1]
  store i32 %initoptions, ptr %initoptions.addr
  call void asm sideeffect "cpuid", "~{ax},~{bx},~{cx},~{dx},~{memory},~{dirflag},~{fpsr},~{flags}"() nounwind
  %1 = call i64 @llvm.readcyclecounter() nounwind ; <i64> [#uses=1]
  %call.i = call i32 @lt_dlinit() nounwind        ; <i32> [#uses=1]
  %tobool.i = icmp ne i32 %call.i, 0              ; <i1> [#uses=1]
  br i1 %tobool.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %entry
  %call1.i = call i32 @warn_dlerror(ptr @.str76843) nounwind ; <i32> [#uses=0]
  store i32 -1, ptr %retval.i
  br label %lt_init.exit

if.end.i:                                         ; preds = %entry
  store i32 0, ptr %retval.i
  br label %lt_init.exit

lt_init.exit:                                     ; preds = %if.end.i, %if.then.i
  %2 = load i32, ptr %retval.i                        ; <i32> [#uses=1]
  call void asm sideeffect "cpuid", "~{ax},~{bx},~{cx},~{dx},~{memory},~{dirflag},~{fpsr},~{flags}"() nounwind
  %3 = call i64 @llvm.readcyclecounter() nounwind ; <i64> [#uses=1]
  %4 = sub i64 %3, %1                             ; <i64> [#uses=1]
  %5 = atomicrmw add ptr getelementptr inbounds ([1216 x i64], ptr @__profiling_callsite_timestamps_live, i32 0, i32 51), i64 %4 monotonic
;CHECK: lock {{xadd|addq}} %rdx, __profiling_callsite_timestamps_live
;CHECK-NEXT: testl [[REG:%e[a-z]+]], [[REG]]
;CHECK-NEXT: jne
  %cmp = icmp eq i32 %2, 0                        ; <i1> [#uses=1]
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %lt_init.exit
  call void @cli_rarload()
  br label %if.end

if.end:                                           ; preds = %if.then, %lt_init.exit
  store i32 0, ptr %retval
  %6 = load i32, ptr %retval                          ; <i32> [#uses=1]
  tail call void asm sideeffect "cpuid", "~{ax},~{bx},~{cx},~{dx},~{memory},~{dirflag},~{fpsr},~{flags}"() nounwind
  %7 = tail call i64 @llvm.readcyclecounter() nounwind ; <i64> [#uses=1]
  %8 = sub i64 %7, %0                             ; <i64> [#uses=1]
  %9 = atomicrmw add ptr getelementptr inbounds ([1216 x i64], ptr @__profiling_callsite_timestamps_live, i32 0, i32 50), i64 %8 monotonic
  ret i32 %6
}

declare void @cli_rarload() nounwind

declare i32 @lt_dlinit()

declare i32 @warn_dlerror(ptr) nounwind

declare i64 @llvm.readcyclecounter() nounwind
