; RUN: llc -mtriple=amdgpu11.00-amd-amdpal < %s | FileCheck %s

; Check that we do not assume any default stack size for PAL code object
; indirect calls. The driver knows the max recursion depth, so it can compute
; a more accurate value.

; CHECK-LABEL: {{^}}test:
; CHECK: ScratchSize: 0
define amdgpu_vs void @test() {
.entry:
  %0 = call i64 @llvm.amdgcn.s.getpc()
  %1 = inttoptr i64 %0 to ptr
  call amdgpu_gfx void %1()
  ret void
}

; PAL cannot grow the stack at runtime, so dynamic stack objects still get the
; default assumed size.

; CHECK-LABEL: {{^}}dynamic_alloca:
; CHECK: ScratchSize: 4112
define amdgpu_cs void @dynamic_alloca(i32 %n, i32 %v) {
.entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %.entry ], [ %i.next, %loop ]
  %a = alloca i32, align 4, addrspace(5)
  store volatile i32 %v, ptr addrspace(5) %a
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK: .cs:
; CHECK: .scratch_memory_size: 0x1010
; CHECK: .vs:
; CHECK: .scratch_memory_size: 0

declare i64 @llvm.amdgcn.s.getpc()
