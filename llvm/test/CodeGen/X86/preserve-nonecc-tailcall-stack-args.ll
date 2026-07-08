; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=static | FileCheck %s

declare preserve_nonecc void @next(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare preserve_nonecc void @next_with_cond(i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define preserve_nonecc void @static_tail_unchanged(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32, i64 %a33, i64 %a34, i64 %a35) {
; CHECK-LABEL: static_tail_unchanged:
; CHECK:       # %bb.0:
; CHECK-NEXT:    jmp next@PLT # TAILCALL
entry:
  musttail call preserve_nonecc void @next(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32, i64 %a33, i64 %a34, i64 %a35)
  ret void
}

define preserve_nonecc void @static_tail_changed(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32, i64 %a33, i64 %a34, i64 %a35) {
; CHECK-LABEL: static_tail_changed:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq 168(%rsp), %r10
; CHECK-NEXT:    incq %r10
; CHECK-NEXT:    movq %r10, 168(%rsp)
; CHECK-NEXT:    jmp next@PLT # TAILCALL
entry:
  %a32x = add i64 %a32, 1
  musttail call preserve_nonecc void @next(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32x, i64 %a33, i64 %a34, i64 %a35)
  ret void
}

define preserve_nonecc void @phi_tail(i1 %cond, i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32, i64 %a33, i64 %a34, i64 %a35) {
; CHECK-LABEL: phi_tail:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq 176(%rsp), %r10
; CHECK-NEXT:    testb $1, %dil
; CHECK-NEXT:    je .LBB2_2
; CHECK:       # %bb.1:
; CHECK-NEXT:    incq %r10
; CHECK:       .LBB2_2:
; CHECK-NEXT:    movq %r10, 176(%rsp)
; CHECK-NEXT:    movzbl %dil, %edi
; CHECK-NEXT:    jmp next_with_cond@PLT # TAILCALL
entry:
  br i1 %cond, label %a, label %b
a:
  %a32x = add i64 %a32, 1
  br label %join
b:
  br label %join
join:
  %a32m = phi i64 [ %a32x, %a ], [ %a32, %b ]
  musttail call preserve_nonecc void @next_with_cond(i1 %cond, i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32m, i64 %a33, i64 %a34, i64 %a35)
  ret void
}
