; -air-portable runs only the passes that define AIR (addAIRPortablePasses):
; LLVM intrinsics and atomics become AIR functions and system values become
; kernel parameters described by !air.kernel. The passes that adapt the module
; to the Metal driver do not run: fneg stays, the device load in the loop is
; not made volatile, the store without a thread index is not guarded, the
; helper is not inlined and the constant buffer is not packed into
; _scalar_buf. The full pipeline does all of these. An ordered atomic, relaxed
; between fences in AIR, keeps its order as agx.atomic.ordering here only.
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s
; RUN: llc -mtriple=air %s -o - | FileCheck %s --check-prefix=METAL
; RUN: not --crash llc -mtriple=air -air-portable -filetype=obj %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=OBJ

; CHECK-LABEL: define internal float @helper(
; CHECK: fneg float
; CHECK-LABEL: define void @kernel(
; CHECK-SAME: ptr addrspace(1) %out, ptr addrspace(2) %scale, <3 x i32> %tid)
; CHECK: call float @air.sqrt.f32(
; CHECK: call float @helper(
; CHECK: call i32 @air.atomic.global.add.s.i32(
; CHECK: call void @air.atomic.fence(i32 3, i32 5, i32 2)
; CHECK-NEXT: call i32 @air.atomic.global.add.s.i32({{.*}}), !agx.atomic.ordering ![[ACQREL:[0-9]+]]
; CHECK-NEXT: call void @air.atomic.fence(i32 3, i32 5, i32 2)
; CHECK-NOT: load volatile
; CHECK: ret void
; CHECK: !air.kernel = !{![[K:[0-9]+]]}
; CHECK-DAG: ![[K]] = !{ptr @kernel,
; CHECK-DAG: ![[ACQREL]] = !{i32 4}

; METAL-LABEL: define void @kernel(
; METAL-SAME: %_scalar_buf
; METAL-NOT: fneg
; METAL-NOT: call float @helper(
; METAL-NOT: agx.atomic.ordering
; METAL: load volatile

; OBJ: -air-portable prints the module; use -filetype=asm

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
declare float @llvm.sqrt.f32(float)

define internal float @helper(float %x) noinline {
  %n = fneg float %x
  ret float %n
}

define void @kernel(ptr addrspace(1) %out, ptr addrspace(2) %scale) "air-kernel" {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tx = extractvalue { i32, i32, i32 } %t, 0
  %s = load float, ptr addrspace(2) %scale
  %r = call float @llvm.sqrt.f32(float %s)
  %h = call float @helper(float %r)
  %old = atomicrmw add ptr addrspace(1) %out, i32 %tx monotonic
  %ord = atomicrmw add ptr addrspace(1) %out, i32 %old acq_rel
  %slot = getelementptr float, ptr addrspace(1) %out, i32 1
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %v = load float, ptr addrspace(1) %slot
  %w = fadd float %v, %h
  store float %w, ptr addrspace(1) %slot
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
