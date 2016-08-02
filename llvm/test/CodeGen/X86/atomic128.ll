; RUN: llc < %s -mtriple=x86_64-apple-macosx10.9 -verify-machineinstrs -mattr=cx16 | FileCheck %s

@var = global i128 0

define i128 @val_compare_and_swap(i128* %p, i128 %oldval, i128 %newval) {
; CHECK-LABEL: val_compare_and_swap:
; CHECK: movq %rsi, %rax
; CHECK: movq %rcx, %rbx
; CHECK: movq %r8, %rcx
; CHECK: lock
; CHECK: cmpxchg16b (%rdi)

  %pair = cmpxchg i128* %p, i128 %oldval, i128 %newval acquire acquire
  %val = extractvalue { i128, i1 } %pair, 0
  ret i128 %val
}

define void @fetch_and_nand(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_nand:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         movq %rdx, %rcx
; CHECK:         andq [[INCHI]], %rcx
; CHECK:         movq %rax, %rbx
  ; INCLO equivalent comes in in %rsi, so it makes sense it stays there.
; CHECK:         andq %rsi, %rbx
; CHECK:         notq %rbx
; CHECK:         notq %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8
  %val = atomicrmw nand i128* %p, i128 %bits release
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_or(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_or:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         movq %rax, %rbx
  ; INCLO equivalent comes in in %rsi, so it makes sense it stays there.
; CHECK:         orq %rsi, %rbx
; CHECK:         movq %rdx, %rcx
; CHECK:         orq [[INCHI]], %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8

  %val = atomicrmw or i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_add(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_add:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         movq %rax, %rbx
  ; INCLO equivalent comes in in %rsi, so it makes sense it stays there.
; CHECK:         addq %rsi, %rbx
; CHECK:         movq %rdx, %rcx
; CHECK:         adcq [[INCHI]], %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8

  %val = atomicrmw add i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_sub(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_sub:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         movq %rax, %rbx
  ; INCLO equivalent comes in in %rsi, so it makes sense it stays there.
; CHECK:         subq %rsi, %rbx
; CHECK:         movq %rdx, %rcx
; CHECK:         sbbq [[INCHI]], %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8

  %val = atomicrmw sub i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_min(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_min:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         cmpq
; CHECK:         sbbq
; CHECK:         setg
; CHECK:         cmovneq %rax, %rbx
; CHECK:         movq [[INCHI]], %rcx
; CHECK:         cmovneq %rdx, %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8

  %val = atomicrmw min i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_max(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_max:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         cmpq
; CHECK:         sbbq
; CHECK:         setge
; CHECK:         cmovneq %rax, %rbx
; CHECK:         movq [[INCHI]], %rcx
; CHECK:         cmovneq %rdx, %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8

  %val = atomicrmw max i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_umin(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_umin:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         cmpq
; CHECK:         sbbq
; CHECK:         seta
; CHECK:         cmovneq %rax, %rbx
; CHECK:         movq [[INCHI]], %rcx
; CHECK:         cmovneq %rdx, %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8

  %val = atomicrmw umin i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define void @fetch_and_umax(i128* %p, i128 %bits) {
; CHECK-LABEL: fetch_and_umax:
; CHECK-DAG:     movq %rdx, [[INCHI:%[a-z0-9]+]]
; CHECK-DAG:     movq (%rdi), %rax
; CHECK-DAG:     movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         cmpq
; CHECK:         sbbq
; CHECK:         setb
; CHECK:         cmovneq %rax, %rbx
; CHECK:         movq [[INCHI]], %rcx
; CHECK:         cmovneq %rdx, %rcx
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

; CHECK:         movq %rax, _var
; CHECK:         movq %rdx, _var+8

  %val = atomicrmw umax i128* %p, i128 %bits seq_cst
  store i128 %val, i128* @var, align 16
  ret void
}

define i128 @atomic_load_seq_cst(i128* %p) {
; CHECK-LABEL: atomic_load_seq_cst:
; CHECK: xorl %eax, %eax
; CHECK: xorl %edx, %edx
; CHECK: xorl %ebx, %ebx
; CHECK: xorl %ecx, %ecx
; CHECK: lock
; CHECK: cmpxchg16b (%rdi)

   %r = load atomic i128, i128* %p seq_cst, align 16
   ret i128 %r
}

define i128 @atomic_load_relaxed(i128* %p) {
; CHECK: atomic_load_relaxed:
; CHECK: xorl %eax, %eax
; CHECK: xorl %edx, %edx
; CHECK: xorl %ebx, %ebx
; CHECK: xorl %ecx, %ecx
; CHECK: lock
; CHECK: cmpxchg16b (%rdi)

   %r = load atomic i128, i128* %p monotonic, align 16
   ret i128 %r
}

define void @atomic_store_seq_cst(i128* %p, i128 %in) {
; CHECK-LABEL: atomic_store_seq_cst:
; CHECK:         movq %rdx, %rcx
; CHECK:         movq %rsi, %rbx
; CHECK:         movq (%rdi), %rax
; CHECK:         movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]
; CHECK-NOT:     callq ___sync_lock_test_and_set_16

   store atomic i128 %in, i128* %p seq_cst, align 16
   ret void
}

define void @atomic_store_release(i128* %p, i128 %in) {
; CHECK-LABEL: atomic_store_release:
; CHECK:         movq %rdx, %rcx
; CHECK:         movq %rsi, %rbx
; CHECK:         movq (%rdi), %rax
; CHECK:         movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

   store atomic i128 %in, i128* %p release, align 16
   ret void
}

define void @atomic_store_relaxed(i128* %p, i128 %in) {
; CHECK-LABEL: atomic_store_relaxed:
; CHECK:         movq %rdx, %rcx
; CHECK:         movq %rsi, %rbx
; CHECK:         movq (%rdi), %rax
; CHECK:         movq 8(%rdi), %rdx

; CHECK: [[LOOP:.?LBB[0-9]+_[0-9]+]]:
; CHECK:         lock
; CHECK:         cmpxchg16b (%rdi)
; CHECK:         jne [[LOOP]]

   store atomic i128 %in, i128* %p unordered, align 16
   ret void
}
