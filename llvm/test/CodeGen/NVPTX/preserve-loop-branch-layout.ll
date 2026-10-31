; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_120 -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=PRESERVE,ACYCLIC
; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_120 -verify-machineinstrs -nvptx-preserve-loop-branch-layout %s -o - | FileCheck %s --check-prefixes=PRESERVE,ACYCLIC
; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_120 -verify-machineinstrs -nvptx-preserve-loop-branch-layout=false %s -o - | FileCheck %s --check-prefixes=DEFAULT,ACYCLIC
; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_100 -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=DEFAULT,ACYCLIC
; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_121 -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=DEFAULT,ACYCLIC
; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_120a -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=DEFAULT,ACYCLIC
; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_120f -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=DEFAULT,ACYCLIC
; RUN: llc -O3 -mtriple=nvptx64 -mcpu=sm_120 -verify-machineinstrs -disable-cgp -nvptx-preserve-loop-branch-layout %s -o - | FileCheck %s --check-prefix=ACYCLIC

; ptxas 13.3 and 13.4 unroll the preserved inner-loop layout four ways for
; sm_120. Reversing the guard/latch branches prevents that unrolling. Keep the
; generic branch optimization passes enabled: the selected loop branches remain
; analyzable, but their true and false targets should not be swapped.
; See https://github.com/llvm/llvm-project/issues/219778.

target triple = "nvptx64-nvidia-cuda"

declare float @llvm.fma.f32(float, float, float)

define ptx_kernel void @nested_dot(ptr addrspace(1) %a, ptr addrspace(1) %b,
                                  ptr addrspace(1) %out, i32 %n, i32 %m) {
; DEFAULT-LABEL: .visible .entry nested_dot(
; DEFAULT: [[EXIT:\$L__BB[0-9]+_[0-9]+]]: // %inner.exit
; DEFAULT: // %outer
; DEFAULT: @!{{%p[0-9]+}} bra [[EXIT]];
; DEFAULT: [[LOOP:\$L__BB[0-9]+_[0-9]+]]: // %loop
; DEFAULT: setp.eq.b32 [[DONE:%p[0-9]+]],
; DEFAULT-NEXT: @![[DONE]] bra [[LOOP]];
; DEFAULT-NEXT: bra.uni [[EXIT]];
;
; PRESERVE-LABEL: .visible .entry nested_dot(
; PRESERVE: [[EXIT:\$L__BB[0-9]+_[0-9]+]]: // %inner.exit
; PRESERVE: // %outer
; PRESERVE: @{{%p[0-9]+}} bra [[PREHEADER:\$L__BB[0-9]+_[0-9]+]];
; PRESERVE-NEXT: bra.uni [[EXIT]];
; PRESERVE-NEXT: [[PREHEADER]]: // %preheader
; PRESERVE: [[LOOP:\$L__BB[0-9]+_[0-9]+]]: // %loop
; PRESERVE: setp.eq.b32 [[DONE:%p[0-9]+]],
; PRESERVE-NEXT: @[[DONE]] bra [[EXIT]];
; PRESERVE-NEXT: bra.uni [[LOOP]];
entry:
  %has.rows = icmp sgt i32 %m, 0
  br i1 %has.rows, label %outer, label %return

outer:
  %row = phi i32 [ 0, %entry ], [ %next.row, %inner.exit ]
  %offset = mul i32 %row, %n
  %has.work = icmp sgt i32 %n, 0
  br i1 %has.work, label %preheader, label %inner.exit

inner.exit:
  %result = phi float [ 0.0, %outer ], [ %sum, %loop ]
  %output = getelementptr float, ptr addrspace(1) %out, i32 %row
  store float %result, ptr addrspace(1) %output, align 4
  %next.row = add nuw nsw i32 %row, 1
  %all.done = icmp eq i32 %next.row, %m
  br i1 %all.done, label %return, label %outer

preheader:
  br label %loop

loop:
  %i = phi i32 [ 0, %preheader ], [ %next, %loop ]
  %acc = phi float [ 0.0, %preheader ], [ %sum, %loop ]
  %index = add i32 %offset, %i
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %index
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %i
  %x = load float, ptr addrspace(1) %pa, align 4
  %y = load float, ptr addrspace(1) %pb, align 4
  %sum = call float @llvm.fma.f32(float %x, float %y, float %acc)
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %inner.exit, label %loop

return:
  ret void
}

; Branch folding should still simplify acyclic control flow with the option on.
define void @acyclic(i32 %x, ptr %out) {
; ACYCLIC-LABEL: .visible .func acyclic(
; ACYCLIC-NOT: bra
; ACYCLIC: ret;
entry:
  %cond = icmp eq i32 %x, 1
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  store float 2.0, ptr %out
  ret void
}
