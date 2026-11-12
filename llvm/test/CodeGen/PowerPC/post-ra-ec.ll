; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.inode.0.12.120 = type { ptr }
%struct.kstat2.1.13.121 = type { i32 }
%struct.task_struct.4.16.124 = type { ptr, %struct.atomic_t.2.14.122, ptr }
%struct.atomic_t.2.14.122 = type { i32 }
%struct.signal_struct.3.15.123 = type { i64 }
%struct.pid.5.17.125 = type { ptr }

; Function Attrs: nounwind
define signext i32 @proc_task_getattr(ptr nocapture readonly %inode, ptr nocapture %stat) #0 {
entry:
  %call1.i = tail call ptr @get_pid_task(ptr undef, i32 zeroext 0) #0
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i64, ptr undef, align 8
  %conv.i = trunc i64 %0 to i32
  %1 = load i32, ptr null, align 4
  %add = add i32 %1, %conv.i
  store i32 %add, ptr null, align 4
  %counter.i.i = getelementptr inbounds %struct.task_struct.4.16.124, ptr %call1.i, i64 0, i32 1, i32 0
  %2 = tail call i32 asm sideeffect "\09lwsync\0A1:\09lwarx\09$0,0,$1\09\09# atomic_dec_return\0A\09addic\09$0,$0,-1\0A\09stwcx.\09$0,0,$1\0A\09bne-\091b\0A\09sync\0A", "=&r,r,~{cr0},~{xer},~{memory}"(ptr %counter.i.i) #0
  %cmp.i = icmp eq i32 %2, 0
  br i1 %cmp.i, label %if.then.i, label %if.end

; CHECK-LABEL: @proc_task_getattr
; CHECK-NOT: stwcx. [[REG:[0-9]+]],0,[[REG]]
; CHECK: blr

if.then.i:                                        ; preds = %if.then
  tail call void @foo(ptr %call1.i) #0
  unreachable

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

declare void @foo(ptr)

declare ptr @get_pid_task(ptr, i32 zeroext)

attributes #0 = { nounwind }

