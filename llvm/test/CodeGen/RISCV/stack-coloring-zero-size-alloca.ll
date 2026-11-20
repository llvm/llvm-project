; RUN: llc -filetype=null %s -mtriple=riscv64 -misched-print-dags 2>&1 | FileCheck %s
; REQUIRES: asserts

;; Test that stack coloring does not corrupt BasicAA results when a slot could
;; be merged into the slot of a zero-sized alloca. Its stack object is one byte,
;; but the alloca, which alias analysis uses, allocates nothing.

define i16 @flip_selected_slot(i1 %c, i8 %v0, i8 %v1) nounwind {
  ; CHECK-LABEL: flip_selected_slot:%bb.{{[0-9]+}} join
  ; CHECK:       SU(0): SB {{.*}}, %stack.1.slot0, 0 :: (store (s8) into %ir.slot0)
  ; CHECK:         Successors:
  ; CHECK-NEXT:      SU(4): Ord  Latency=0 Memory
  ; CHECK-NEXT:      SU(5): Ord  Latency=1 Memory
  ; CHECK-NEXT:      SU(2): Ord  Latency=1 Memory
  ; CHECK:       SU(2): {{.*}} = LBU {{.*}} :: (load (s8) from %ir.p)
  ; CHECK:       SU(4): SB {{.*}} :: (store (s8) into %ir.p)
  ; CHECK:         Successors:
  ; CHECK-NEXT:      SU(5): Ord  Latency=1 Memory
  ; CHECK-NEXT:      SU(6): Ord  Latency=1 Memory
  ; CHECK:       SU(5): {{.*}} = LBU %stack.1.slot0, 0 :: (dereferenceable load (s8) from %ir.slot0)
entry:
  %empty = alloca [0 x i8], align 1
  %slot0 = alloca i8, align 1
  %slot1 = alloca i8, align 1
  ;; Unused; makes the frame large enough for stack coloring to run.
  %pad = alloca [16 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr %empty)
  call void @use(ptr %empty)
  call void @llvm.lifetime.end.p0(ptr %empty)
  call void @llvm.lifetime.start.p0(ptr %slot0)
  call void @llvm.lifetime.start.p0(ptr %slot1)
  br i1 %c, label %pick0, label %pick1

pick0:
  br label %join

pick1:
  br label %join

join:
  %p = phi ptr [ %slot0, %pick0 ], [ %slot1, %pick1 ]
  ;; %p can point to %slot0, so the store to %slot0 must not be reordered past
  ;; the load or store through %p, and the reload of %slot0 must stay after the
  ;; store through %p.
  store i8 %v0, ptr %slot0, align 1
  store i8 %v1, ptr %slot1, align 1
  %old = load i8, ptr %p, align 1
  %new = xor i8 %old, 1
  store i8 %new, ptr %p, align 1
  %r0 = load i8, ptr %slot0, align 1
  %r1 = load i8, ptr %slot1, align 1
  call void @llvm.lifetime.end.p0(ptr %slot0)
  call void @llvm.lifetime.end.p0(ptr %slot1)
  %z0 = zext i8 %r0 to i16
  %z1 = zext i8 %r1 to i16
  %hi = shl i16 %z1, 8
  %res = or i16 %hi, %z0
  ret i16 %res
}

declare void @use(ptr)
