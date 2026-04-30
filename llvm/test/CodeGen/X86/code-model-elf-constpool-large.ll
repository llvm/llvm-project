; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -o %t
; RUN: llvm-readelf -S %t | FileCheck %s

; Verify that anonymous constant pool entries get SHF_X86_64_LARGE
; and are placed in .lrodata.cst* sections under the large code model.

; CHECK: .lrodata.cst16 {{.*}} AMl
; CHECK: .lrodata.cst4  {{.*}} AMl

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

define void @vec_add(ptr %out, ptr %a, ptr %b, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %idx = sext i32 %iv to i64
  %pa = getelementptr float, ptr %a, i64 %idx
  %pb = getelementptr float, ptr %b, i64 %idx
  %po = getelementptr float, ptr %out, i64 %idx
  %va = load <4 x float>, ptr %pa, align 4
  %vb = load <4 x float>, ptr %pb, align 4
  %add = fadd <4 x float> %va, %vb
  ; Adding a vector splat of 1.0 forces a constant pool entry in .rodata.cst16
  %ones = fadd <4 x float> %add, <float 1.0, float 1.0, float 1.0, float 1.0>
  store <4 x float> %ones, ptr %po, align 4
  %iv.next = add i32 %iv, 4
  %done = icmp sge i32 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A scalar float constant forces a .rodata.cst4 entry.
define float @scalar_const(float %x) {
  %r = fadd float %x, 1.0
  ret float %r
}
