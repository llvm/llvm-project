; RUN: llc -mtriple=bpfel -mcpu=v4 -gotol-abs-low-bound 0 -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; Source:
;   // This test covers all three cases:
;   //   (1). jmp to another basic block (not the follow-through one)
;   //   (2). conditional jmp (follow-through and non-follow-through)
;   //   (3). conditional jmp followed by an unconditional jmp
;   // To trigger case (3) the following code is developed which
;   // covers case (1) and (2) as well.
;   unsigned foo(unsigned a, unsigned b) {
;     unsigned s = b;
;     if (a < b)
;       goto next;
;     else
;       goto next2;
;   begin:
;     s /= b;
;     if (s > a)
;       return s * s;
;   next:
;     s *= a;
;     if (s > b)
;       goto begin;
;   next2:
;     s *= b;
;     if (s > a)
;       goto begin;
;     return s;
;   }
; Compilation flags:
;   clang -target bpf -O2 -mcpu=v4 -S -emit-llvm t.c

; Function Attrs: nofree norecurse nosync nounwind memory(none)
define dso_local i32 @foo(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 {
entry:
  %cmp = icmp ult i32 %a, %b
  br i1 %cmp, label %next, label %next2

; case (3): conditional jmp followed by an unconditional jmp
; CHECK:        w0 = w2
; CHECK-NEXT:   if w1 < w2 goto
; CHECK:        gotol LBB0_4    # encoding: [0x06'A',A,A,A,0x00,0x00,0x00,0x00]
; CHECK-NEXT:                   # fixup A - offset: 0, value: LBB0_4, kind: FK_BPF_PCRel_4

begin:                                            ; preds = %next2, %next
  %s.0 = phi i32 [ %mul3, %next ], [ %mul7, %next2 ]
  %div = udiv i32 %s.0, %b
  %cmp1 = icmp ugt i32 %div, %a
  br i1 %cmp1, label %if.then2, label %next

; case (2): conditional jmp
; CHECK:        w0 *= w1
; CHECK-NEXT:   if w0 > w2 goto LBB0_7
; CHECK:        goto LBB0_4
; CHECK-LABEL:  LBB0_7:
; CHECK:        gotol

; CHECK-LABEL:  LBB0_4:

if.then2:                                         ; preds = %begin
  %mul = mul i32 %div, %div
  br label %cleanup

; case (1): unconditional jmp
; CHECK:        w0 *= w0
; CHECK-NEXT:   gotol

next:                                             ; preds = %begin, %entry
  %s.1 = phi i32 [ %b, %entry ], [ %div, %begin ]
  %mul3 = mul i32 %s.1, %a
  %cmp4 = icmp ugt i32 %mul3, %b
  br i1 %cmp4, label %begin, label %next2

next2:                                            ; preds = %next, %entry
  %s.2 = phi i32 [ %mul3, %next ], [ %b, %entry ]
  %mul7 = mul i32 %s.2, %b
  %cmp8 = icmp ugt i32 %mul7, %a
  br i1 %cmp8, label %begin, label %cleanup

cleanup:                                          ; preds = %next2, %if.then2
  %retval.0 = phi i32 [ %mul, %if.then2 ], [ %mul7, %next2 ]
  ret i32 %retval.0
}

attributes #0 = { nofree norecurse nosync nounwind memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v4" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 18.0.0 (https://github.com/llvm/llvm-project.git dccf0f74657ce8c50eb1e997bae356c32d7b1ffe)"}
