; RUN: llc -mtriple=nvptx64 -mcpu=sm_90 -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc -mtriple=nvptx64 -mcpu=sm_90 -O2 -enable-tail-merge -tail-merge-size=1 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,MERGE

; The side-exit PHI coalesces two different uniform predicates. With test=true,
; a=0 and b=1, they differ between the half-warps after their branches merge.
; CHECK-LABEL: .visible .entry separate_uniform_predicates(
; DEFAULT: // left [[P:%p[0-9]+]]
; DEFAULT: @{{!?}}[[P]] bra.uni
; DEFAULT: // right [[P]]
; DEFAULT: @{{!?}}[[P]] bra.uni
; MERGE: // left [[P:%p[0-9]+]]
; MERGE: // right [[P]]
; MERGE: @{{!?}}[[P]] bra{{[ \t]}}

define ptx_kernel void @separate_uniform_predicates(i1 %test, i32 %a, i32 %b) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %split = icmp ult i32 %lane, 16
  br i1 %split, label %left, label %right
left:
  %p = icmp eq i32 %a, 0
  call void asm sideeffect "// left $0", "b"(i1 %p)
  br i1 %test, label %lefttest, label %sideexit
right:
  %q = icmp eq i32 %b, 0
  call void asm sideeffect "// right $0", "b"(i1 %q)
  br i1 %test, label %righttest, label %sideexit
lefttest:
  br i1 %p, label %true, label %false
righttest:
  br i1 %q, label %true, label %false
sideexit:
  %r = phi i1 [%p, %left], [%q, %right]
  call void asm sideeffect "// result $0", "b"(i1 %r)
  ret void
true:
  call void asm sideeffect "// true", ""()
  ret void
false:
  call void asm sideeffect "// false", ""()
  ret void
}

; The adjacent fold keeps an existing path to the retained branch. Its predicate
; stays uniform within each loop iteration.
; CHECK-LABEL: .visible .entry adjacent_loop(
; CHECK: setp.eq.b32 [[ITER_PRED:%p[0-9]+]], {{%r[0-9]+}}, 0;
; CHECK: @{{!?}}[[ITER_PRED]] bra.uni
define ptx_kernel void @adjacent_loop(ptr addrspace(1) %out, i1 %flip, i32 %limit) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %side = icmp ult i32 %tid, 16
  %ptr = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  %p = icmp eq i32 %i, 0
  %q = xor i1 %p, %flip
  br i1 %side, label %a, label %c
a:
  br i1 %p, label %true, label %b
b:
  br i1 %p, label %true, label %false
false:
  store volatile i32 %i, ptr addrspace(1) %ptr
  %stop = icmp sge i32 %i, %limit
  br i1 %stop, label %exit, label %latch
c:
  br i1 %q, label %latch, label %b
latch:
  %next = add i32 %i, 1
  br label %header
true:
  store volatile i32 99, ptr addrspace(1) %ptr
  ret void
exit:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
