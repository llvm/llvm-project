; RUN: llc -mtriple=riscv32 < %s | FileCheck %s -check-prefix=RV32I
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv32 -mattr=+save-restore < %s | FileCheck %s -check-prefix=RV32I-SR
; RUN: llc -mtriple=riscv64 -mattr=+save-restore < %s | FileCheck %s -check-prefix=RV64I-SR
; RUN: llc -mtriple=riscv32 -mattr=+f,+save-restore -target-abi=ilp32f < %s | FileCheck %s -check-prefix=RV32I-FP-SR
; RUN: llc -mtriple=riscv64 -mattr=+f,+d,+save-restore -target-abi=lp64d < %s | FileCheck %s -check-prefix=RV64I-FP-SR

; Check that the correct save/restore libcalls are generated.

@var0 = global [18 x i32] zeroinitializer
@var1 = global [24 x i32] zeroinitializer
@var2 = global [30 x i32] zeroinitializer

define void @callee_saved0() nounwind {
; RV32I-LABEL: callee_saved0:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: callee_saved0:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: callee_saved0:
; RV32I-SR:         call t0, __riscv_save_4
; RV32I-SR:         tail __riscv_restore_4
;
; RV64I-SR-LABEL: callee_saved0:
; RV64I-SR:         call t0, __riscv_save_4
; RV64I-SR:         tail __riscv_restore_4
;
; RV32I-FP-SR-LABEL: callee_saved0:
; RV32I-FP-SR:         call t0, __riscv_save_4
; RV32I-FP-SR:         tail __riscv_restore_4
;
; RV64I-FP-SR-LABEL: callee_saved0:
; RV64I-FP-SR:         call t0, __riscv_save_4
; RV64I-FP-SR:         tail __riscv_restore_4
  %val = load [18 x i32], ptr @var0
  store volatile [18 x i32] %val, ptr @var0
  ret void
}

define void @callee_saved1() nounwind {
; RV32I-LABEL: callee_saved1:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: callee_saved1:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: callee_saved1:
; RV32I-SR:         call t0, __riscv_save_10
; RV32I-SR:         tail __riscv_restore_10
;
; RV64I-SR-LABEL: callee_saved1:
; RV64I-SR:         call t0, __riscv_save_10
; RV64I-SR:         tail __riscv_restore_10
;
; RV32I-FP-SR-LABEL: callee_saved1:
; RV32I-FP-SR:         call t0, __riscv_save_10
; RV32I-FP-SR:         tail __riscv_restore_10
;
; RV64I-FP-SR-LABEL: callee_saved1:
; RV64I-FP-SR:         call t0, __riscv_save_10
; RV64I-FP-SR:         tail __riscv_restore_10
  %val = load [24 x i32], ptr @var1
  store volatile [24 x i32] %val, ptr @var1
  ret void
}

define void @callee_saved2() nounwind {
; RV32I-LABEL: callee_saved2:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: callee_saved2:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: callee_saved2:
; RV32I-SR:         call t0, __riscv_save_12
; RV32I-SR:         tail __riscv_restore_12
;
; RV64I-SR-LABEL: callee_saved2:
; RV64I-SR:         call t0, __riscv_save_12
; RV64I-SR:         tail __riscv_restore_12
;
; RV32I-FP-SR-LABEL: callee_saved2:
; RV32I-FP-SR:         call t0, __riscv_save_12
; RV32I-FP-SR:         tail __riscv_restore_12
;
; RV64I-FP-SR-LABEL: callee_saved2:
; RV64I-FP-SR:         call t0, __riscv_save_12
; RV64I-FP-SR:         tail __riscv_restore_12
  %val = load [30 x i32], ptr @var2
  store volatile [30 x i32] %val, ptr @var2
  ret void
}

; Check that floating point callee saved registers are still manually saved and
; restored.

define void @callee_saved_fp() nounwind {
; RV32I-LABEL: callee_saved_fp:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: callee_saved_fp:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: callee_saved_fp:
; RV32I-SR:         call t0, __riscv_save_7
; RV32I-SR:         tail __riscv_restore_7
;
; RV64I-SR-LABEL: callee_saved_fp:
; RV64I-SR:         call t0, __riscv_save_7
; RV64I-SR:         tail __riscv_restore_7
;
; RV32I-FP-SR-LABEL: callee_saved_fp:
; RV32I-FP-SR:         call t0, __riscv_save_7
; RV32I-FP-SR-NEXT:    addi sp, sp, -16
; RV32I-FP-SR-NEXT:    fsw fs0, 12(sp)
; RV32I-FP-SR:         flw fs0, 12(sp)
; RV32I-FP-SR-NEXT:    addi sp, sp, 16
; RV32I-FP-SR-NEXT:    tail __riscv_restore_7
;
; RV64I-FP-SR-LABEL: callee_saved_fp:
; RV64I-FP-SR:         call t0, __riscv_save_7
; RV64I-FP-SR-NEXT:    addi sp, sp, -16
; RV64I-FP-SR-NEXT:    fsd fs0, 8(sp)
; RV64I-FP-SR:         fld fs0, 8(sp)
; RV64I-FP-SR-NEXT:    addi sp, sp, 16
; RV64I-FP-SR-NEXT:    tail __riscv_restore_7
  call void asm sideeffect "", "~{f8},~{x9},~{x18},~{x19},~{x20},~{x21},~{x22}"()
  ret void
}

; Check that preserving tail calls is preferred over save/restore

declare i32 @tail_callee(i32 %i)

define i32 @tail_call(i32 %i) nounwind {
; RV32I-LABEL: tail_call:
; RV32I-NOT:     call t0, __riscv_save
; RV32I:         tail tail_callee
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: tail_call:
; RV64I-NOT:     call t0, __riscv_save
; RV64I:         tail tail_callee
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: tail_call:
; RV32I-SR-NOT:     call t0, __riscv_save
; RV32I-SR:         tail tail_callee
; RV32I-SR-NOT:     tail __riscv_restore
;
; RV64I-SR-LABEL: tail_call:
; RV64I-SR-NOT:     call t0, __riscv_save
; RV64I-SR:         tail tail_callee
; RV64I-SR-NOT:     tail __riscv_restore
;
; RV32I-FP-SR-LABEL: tail_call:
; RV32I-FP-SR-NOT:     call t0, __riscv_save
; RV32I-FP-SR:         tail tail_callee
; RV32I-FP-SR-NOT:     tail __riscv_restore
;
; RV64I-FP-SR-LABEL: tail_call:
; RV64I-FP-SR-NOT:     call t0, __riscv_save
; RV64I-FP-SR:         tail tail_callee
; RV64I-FP-SR-NOT:     tail __riscv_restore
entry:
  %val = load [18 x i32], ptr @var0
  store volatile [18 x i32] %val, ptr @var0
  %r = tail call i32 @tail_callee(i32 %i)
  ret i32 %r
}

; Check that functions with varargs do not use save/restore code

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define i32 @varargs(ptr %fmt, ...) nounwind {
; RV32I-LABEL: varargs:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: varargs:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: varargs:
; RV32I-SR-NOT:     call t0, __riscv_save
; RV32I-SR-NOT:     tail __riscv_restore
;
; RV64I-SR-LABEL: varargs:
; RV64I-SR-NOT:     call t0, __riscv_save
; RV64I-SR-NOT:     tail __riscv_restore
;
; RV32I-FP-SR-LABEL: varargs:
; RV32I-FP-SR-NOT:     call t0, __riscv_save
; RV32I-FP-SR-NOT:     tail __riscv_restore
;
; RV64I-FP-SR-LABEL: varargs:
; RV64I-FP-SR-NOT:     call t0, __riscv_save
; RV64I-FP-SR-NOT:     tail __riscv_restore
  %va = alloca ptr, align 4
  call void @llvm.va_start(ptr %va)
  %argp.cur = load ptr, ptr %va, align 4
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  store ptr %argp.next, ptr %va, align 4
  %1 = load i32, ptr %argp.cur, align 4
  call void @llvm.va_end(ptr %va)
  ret i32 %1
}

define void @many_args(i32, i32, i32, i32, i32, i32, i32, i32, i32) nounwind {
; RV32I-LABEL: many_args:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: many_args:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: many_args:
; RV32I-SR:         call t0, __riscv_save_4
; RV32I-SR:         tail __riscv_restore_4
;
; RV64I-SR-LABEL: many_args:
; RV64I-SR:         call t0, __riscv_save_4
; RV64I-SR:         tail __riscv_restore_4
;
; RV32I-FP-SR-LABEL: many_args:
; RV32I-FP-SR:         call t0, __riscv_save_4
; RV32I-FP-SR:         tail __riscv_restore_4
;
; RV64I-FP-SR-LABEL: many_args:
; RV64I-FP-SR:         call t0, __riscv_save_4
; RV64I-FP-SR:         tail __riscv_restore_4
entry:
  %val = load [18 x i32], ptr @var0
  store volatile [18 x i32] %val, ptr @var0
  ret void
}

; Check that dynamic allocation calculations remain correct

declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)
declare void @notdead(ptr)

define void @alloca(i32 %n) nounwind {
; RV32I-LABEL: alloca:
; RV32I-NOT:     call t0, __riscv_save
; RV32I:         addi s0, sp, 16
; RV32I:         addi sp, s0, -16
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: alloca:
; RV64I-NOT:     call t0, __riscv_save
; RV64I:         addi s0, sp, 32
; RV64I:         addi sp, s0, -32
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: alloca:
; RV32I-SR:         call t0, __riscv_save_2
; RV32I-SR:         addi s0, sp, 16
; RV32I-SR:         addi sp, s0, -16
; RV32I-SR:         tail __riscv_restore_2
;
; RV64I-SR-LABEL: alloca:
; RV64I-SR:         call t0, __riscv_save_2
; RV64I-SR:         addi s0, sp, 32
; RV64I-SR:         addi sp, s0, -32
; RV64I-SR:         tail __riscv_restore_2
;
; RV32I-FP-SR-LABEL: alloca:
; RV32I-FP-SR:         call t0, __riscv_save_2
; RV32I-FP-SR:         addi s0, sp, 16
; RV32I-FP-SR:         addi sp, s0, -16
; RV32I-FP-SR:         tail __riscv_restore_2
;
; RV64I-FP-SR-LABEL: alloca:
; RV64I-FP-SR:         call t0, __riscv_save_2
; RV64I-FP-SR:         addi s0, sp, 32
; RV64I-FP-SR:         addi sp, s0, -32
; RV64I-FP-SR:         tail __riscv_restore_2
  %sp = call ptr @llvm.stacksave()
  %addr = alloca i8, i32 %n
  call void @notdead(ptr %addr)
  call void @llvm.stackrestore(ptr %sp)
  ret void
}

; Check that functions with interrupt attribute do not use save/restore code

declare i32 @foo(...)
define void @interrupt() nounwind "interrupt"="supervisor" {
; RV32I-LABEL: interrupt:
; RV32I-NOT:     call t0, __riscv_save
; RV32I-NOT:     tail __riscv_restore
;
; RV64I-LABEL: interrupt:
; RV64I-NOT:     call t0, __riscv_save
; RV64I-NOT:     tail __riscv_restore
;
; RV32I-SR-LABEL: interrupt:
; RV32I-SR-NOT:     call t0, __riscv_save
; RV32I-SR-NOT:     tail __riscv_restore
;
; RV64I-SR-LABEL: interrupt:
; RV64I-SR-NOT:     call t0, __riscv_save
; RV64I-SR-NOT:     tail __riscv_restore
;
; RV32I-FP-SR-LABEL: interrupt:
; RV32I-FP-SR-NOT:     call t0, __riscv_save
; RV32I-FP-SR-NOT:     tail __riscv_restore
;
; RV64I-FP-SR-LABEL: interrupt:
; RV64I-FP-SR-NOT:     call t0, __riscv_save
; RV64I-FP-SR-NOT:     tail __riscv_restore
  %call = call i32 @foo()
  ret void
}
