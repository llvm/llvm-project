; RUN: llc < %s -mtriple=sparc   -relocation-model=pic | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic | FileCheck %s --check-prefix=V9
; RUN: llc < %s -mtriple=sparc   -relocation-model=pic -stop-after=finalize-isel | \
; RUN:   FileCheck %s --check-prefix=MIR-V8
; RUN: llc < %s -mtriple=sparcv9 -relocation-model=pic -stop-after=finalize-isel | \
; RUN:   FileCheck %s --check-prefix=MIR-V9

;; The initial-exec TLS offset is loaded from the GOT, which is constant, so
;; the load must carry a GOT memory operand and be hoisted out of the loop.

; MIR-V8: TLS_LDrr {{.*}} @t :: (load (s32) from got)
; MIR-V9: TLS_LDXrr {{.*}} @t :: (load (s64) from got)

; V8-LABEL: loop:
; V8:         ld [{{%[gilo][0-7]}}+{{%[gilo][0-7]}}], {{%[gilo][0-7]}}, %tie_ld(t)
; V8:       .LBB0_{{[0-9]+}}:
; V8-NOT:     %tie_ld(t)
; V8:       .Lfunc_end0:

; V9-LABEL: loop:
; V9:         ldx [{{%[gilo][0-7]}}+{{%[gilo][0-7]}}], {{%[gilo][0-7]}}, %tie_ldx(t)
; V9:       .LBB0_{{[0-9]+}}:
; V9-NOT:     %tie_ldx(t)
; V9:       .Lfunc_end0:

@t = external thread_local(initialexec) global i32
declare void @use(i32)

define void @loop(i32 %n) {
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %p = call ptr @llvm.threadlocal.address.p0(ptr @t)
  %v = load i32, ptr %p
  %c = icmp eq i32 %v, %i
  br i1 %c, label %call, label %latch
call:
  %q = call ptr @llvm.threadlocal.address.p0(ptr @t)
  store i32 %i, ptr %q
  call void @use(i32 %i)
  br label %latch
latch:
  %inc = add i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %body
exit:
  ret void
}
