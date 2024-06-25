; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo


define void @fence_instrs() {
  fence acquire
  fence release
  fence acq_rel
  fence seq_cst

  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") release
  fence syncscope("singlethread") acq_rel
  fence syncscope("singlethread") seq_cst

  ret void
}

define void @atomic_load_store(ptr %word) {
  ; Test different atomic loads
  %ld.1 = load atomic i32, ptr %word monotonic, align 4
  %ld.2 = load atomic volatile i32, ptr %word acquire, align 4
  %ld.3 = load atomic volatile i32, ptr %word seq_cst, align 4
  %ld.4 = load atomic volatile i32, ptr %word syncscope("singlethread") acquire, align 4
  %ld.5 = load atomic volatile i32, ptr %word syncscope("singlethread") seq_cst, align 4
  %ld.6 = load atomic i32, ptr %word syncscope("singlethread") seq_cst, align 4

  ; Test different atomic stores
  store atomic i32 1, ptr %word monotonic, align 4
  store atomic volatile i32 2, ptr %word release, align 4
  store atomic volatile i32 3, ptr %word seq_cst, align 4
  store atomic volatile i32 4, ptr %word syncscope("singlethread") release, align 4
  store atomic volatile i32 5, ptr %word syncscope("singlethread") seq_cst, align 4
  store atomic i32 6, ptr %word syncscope("singlethread") seq_cst, align 4
  ret void
}

define void @atomic_rmw_ops(ptr %p, i32 %i, float %f) {
  ; Test all atomicrmw operations
  %a.xchg      = atomicrmw xchg      ptr %p, i32 %i acq_rel, align 8
  %a.add       = atomicrmw add       ptr %p, i32 %i acq_rel, align 8
  %a.sub       = atomicrmw sub       ptr %p, i32 %i acq_rel, align 8
  %a.and       = atomicrmw and       ptr %p, i32 %i acq_rel, align 8
  %a.nand      = atomicrmw nand      ptr %p, i32 %i acq_rel, align 8
  %a.or        = atomicrmw or        ptr %p, i32 %i acq_rel, align 8
  %a.xor       = atomicrmw xor       ptr %p, i32 %i acq_rel, align 8
  %a.max       = atomicrmw max       ptr %p, i32 %i acq_rel, align 8
  %a.min       = atomicrmw min       ptr %p, i32 %i acq_rel, align 8
  %a.umax      = atomicrmw umax      ptr %p, i32 %i acq_rel, align 8
  %a.umin      = atomicrmw umin      ptr %p, i32 %i acq_rel, align 8

  %a.fadd      = atomicrmw fadd      ptr %p, float %f acq_rel, align 8
  %a.fsub      = atomicrmw fsub      ptr %p, float %f acq_rel, align 8
  %a.fmax      = atomicrmw fmax      ptr %p, float %f acq_rel, align 8
  %a.fmin      = atomicrmw fmin      ptr %p, float %f acq_rel, align 8

  %a.uinc_wrap = atomicrmw uinc_wrap ptr %p, i32 %i acq_rel, align 8
  %a.udec_wrap = atomicrmw udec_wrap ptr %p, i32 %i acq_rel, align 8

  ret void
}

define i32 @main() {
  %1 = alloca i32, align 4
  %2 = cmpxchg ptr %1, i32 2, i32 3 seq_cst acquire
  %3 = extractvalue { i32, i1 } %2, 0
  ret i32 %3
}
