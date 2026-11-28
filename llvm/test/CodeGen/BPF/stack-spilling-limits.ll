; RUN: split-file %s %t
; RUN: llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=64 < %t/exact.ll | FileCheck %s --check-prefix=EXACT
; RUN: not llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=64 < %t/overflow.ll 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=64 < %t/args-overflow.ll 2>&1 | FileCheck %s --check-prefix=ARGS
; RUN: not llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=64 < %t/call-overflow.ll 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=64 < %t/spill-overflow.ll 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not --crash llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=0 < %t/exact.ll 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not --crash llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=65 < %t/exact.ll 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not --crash llc -mtriple=bpfel -bpf-stack-spilling -bpf-stack-size=32832 < %t/exact.ll 2>&1 | FileCheck %s --check-prefix=INVALID

; EXACT-LABEL: exact:
; EXACT-DAG: r{{[0-9]+}} = *(u64 *)(r10 - 64)
; EXACT-DAG: *(u64 *)(r10 - 56) = r{{[0-9]+}}
; EXACT: exit
; OVERFLOW: error: {{.*}}BPF stack frame exceeds the stack size limit
; ARGS: error: {{.*}}BPF stack frame exceeds the stack size limit
; CALL: error: {{.*}}stack arguments exceed the BPF stack size limit
; INVALID: LLVM ERROR: BPF stack spilling requires a stack size that is a multiple of 64 between 64 and 32768 bytes

;--- exact.ll
; Incoming arguments and locals exactly fill the frame. Outgoing arguments
; must not count towards this frame's size.
declare void @ext(ptr, i64, i64, i64, i64, i64)
define i64 @exact(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %x) {
  %local = alloca [56 x i8], align 8
  store volatile i64 %x, ptr %local
  call void @ext(ptr %local, i64 2, i64 3, i64 4, i64 5, i64 %x)
  ret i64 0
}

;--- overflow.ll
; The local alone fits, but the incoming argument makes the frame overflow.
define i64 @overflow(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %x) {
  %local = alloca [64 x i8], align 8
  store volatile i64 %x, ptr %local
  ret i64 %x
}

;--- args-overflow.ll
; Even unused incoming arguments must fit in the frame.
define void @args_overflow(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
  ret void
}

;--- call-overflow.ll
declare void @ext(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
define void @call_overflow() {
  call void @ext(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  ret void
}

;--- spill-overflow.ll
; Late caller-saved spills for bpf_fastcall must also respect the frame limit.
declare i32 @ext() "bpf_fastcall"
define i32 @spill_overflow(i32 %a) {
  %local = alloca [64 x i8], align 8
  store volatile i32 %a, ptr %local
  %r = call i32 @ext() "bpf_fastcall"
  ret i32 %a
}
