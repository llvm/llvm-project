; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack -stop-after=finalize-isel < %s \
; RUN:   | FileCheck --check-prefix=MIR %s
; RUN: opt -mtriple=bpfel -passes=bpf-arena-stack -S < %s | FileCheck --check-prefix=IR %s
; RUN: not llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack -o /dev/null %S/Inputs/arena-stack-err.ll 2>&1 \
; RUN:   | FileCheck --check-prefix=ERR %s

; Locals are in the stack in arena. It grows up: r12 is to the frame there what
; r10 is to the frame in the stack of the program.

define void @take(ptr %p) noinline {
  ret void
}
declare i64 @bpf_kfunc(ptr)

; The first insn makes the frame. There is nothing to undo.
define i64 @direct(i64 %v) {
; CHECK-LABEL: direct:
; CHECK:       # %bb.0:
; CHECK-NEXT:    r12 += 32
; CHECK:         *(u64 *)(r12 - 16) = r1
; CHECK:         r1 = r12
; CHECK-NEXT:    r1 += -24
; CHECK-NEXT:    call take
; CHECK:         r0 = *(u64 *)(r12 - 16)
; CHECK-NEXT:    exit
; MIR-LABEL: name: direct
; MIR:       stack:
; MIR:         - { id: 0, name: a, {{.*}}size: 24, alignment: 8,
; MIR-NEXT:        stack-id: bpf-arena
; IR-LABEL: @direct
; IR: %a = alloca [3 x i64], align 8, !bpf.arena.stack
  %a = alloca [3 x i64], align 8
  %p = getelementptr i8, ptr %a, i64 8
  store i64 %v, ptr %p
  call void @take(ptr %a)
  %r = load i64, ptr %p
  ret i64 %r
}

; The address of a local is a number: r12 plus the offset.
define void @address() {
; CHECK-LABEL: address:
; CHECK:         r12 += 16
; CHECK-NEXT:    r1 = r12
; CHECK-NEXT:    r1 += -16
; CHECK-NEXT:    call take
; CHECK-NEXT:    r1 = r12
; CHECK-NEXT:    r1 += -8
; CHECK-NEXT:    call take
  %a = alloca i64, align 8
  %b = alloca i64, align 8
  call void @take(ptr %b)
  call void @take(ptr %a)
  ret void
}

; An offset that is not a constant: the address is cast like any other.
define i64 @variable(i64 %i) {
; CHECK-LABEL: variable:
; CHECK:         r12 += 64
; CHECK:         [[A:r[0-9]]] = r12
; CHECK-NEXT:    [[A]] += -64
; CHECK:         [[K:r[0-9]]] = addr_space_cast([[A]], 0, 1)
; CHECK:         [[K]] += r{{[0-9]}}
; CHECK:         r0 = *(u64 *)([[K]] + 0)
  %a = alloca [8 x i64], align 8
  call void @take(ptr %a)
  %p = getelementptr i64, ptr %a, i64 %i
  %r = load i64, ptr %p
  ret i64 %r
}

; The kernel doesn't take atomics through r12.
define void @atomic() {
; CHECK-LABEL: atomic:
; CHECK:         [[A:r[0-9]]] = r12
; CHECK:         [[K:r[0-9]]] = addr_space_cast([[A]], 0, 1)
; CHECK:         atomic_fetch_add((u64 *)([[K]] + 0), r{{[0-9]}})
  %a = alloca i64, align 8
  call void @take(ptr %a)
  %old = atomicrmw add ptr %a, i64 1 seq_cst
  ret void
}

; memset is stores.
define void @set() {
; CHECK-LABEL: set:
; CHECK:         r12 += 16
; CHECK-DAG:     *(u64 *)(r12 - 16) = 0
; CHECK-DAG:     *(u64 *)(r12 - 8) = 0
; CHECK-NOT:     addr_space_cast
; CHECK:         exit
  %a = alloca [2 x i64], align 8
  call void @llvm.memset.p0.i64(ptr %a, i8 0, i64 16, i1 false)
  call void @take(ptr %a)
  ret void
}

; Helpers and kfuncs take memory that the kernel knows about.
define i64 @kernel() {
; CHECK-LABEL: kernel:
; CHECK-NOT:     r12
; CHECK:         r1 = r10
; CHECK-NEXT:    r1 += -8
; CHECK-NEXT:    call bpf_kfunc
; IR-LABEL: @kernel
; IR: %a = alloca i64, align 8{{$}}
  %a = alloca i64, align 8
  %r = call i64 @bpf_kfunc(ptr %a)
  ret i64 %r
}

; The size of the frame is a multiple of 16, so that r12 stays aligned.
define void @align16() {
; CHECK-LABEL: align16:
; CHECK:         r12 += 48
; CHECK-NEXT:    r1 = r12
; CHECK-NEXT:    r1 += -1
; CHECK-NEXT:    call take
; CHECK-NEXT:    r1 = r12
; CHECK-NEXT:    r1 += -33
; CHECK-NEXT:    call take
; CHECK-NEXT:    r1 = r12
; CHECK-NEXT:    r1 += -32
; CHECK-NEXT:    call take
  %a = alloca i8, align 1
  %b = alloca i128, align 16
  %c = alloca i8, align 1
  call void @take(ptr %a)
  call void @take(ptr %c)
  call void @take(ptr %b)
  ret void
}

; Nothing in arena, no frame. Spilled registers are on the stack of the program.
define i64 @none(i64 %v) {
; CHECK-LABEL: none:
; CHECK-NOT:     r12
; CHECK:         exit
  ret i64 %v
}

; ERR: error: {{.*}}in function large {{.*}}: locals don't fit into the stack in arena
