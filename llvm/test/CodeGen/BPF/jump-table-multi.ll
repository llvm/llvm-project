; RUN: llc -O2 -bpf-min-jump-table-entries=1 -mtriple=bpfel -mcpu=v4 < %s | FileCheck %s

; Check that two jump tables of different size are generated

define i64 @foo(i64 %v1, i64 %v2) {
; CHECK:      .LBPF.JX.0.0:
; CHECK-NEXT:        .reloc 0, FK_SecRel_8, BPF.JT.0.0
; CHECK-NEXT:        gotox r1

; CHECK:      .LBPF.JX.0.1:
; CHECK-NEXT:        .reloc 0, FK_SecRel_8, BPF.JT.0.1
; CHECK-NEXT:        gotox r2

; CHECK:        .section        .jumptables,"",@progbits

; CHECK-NEXT: [[m1:.*]] = (({{.*}}-.LBPF.JX.0.0)>>3)-1
; CHECK-NEXT: [[m2:.*]] = (({{.*}}-.LBPF.JX.0.0)>>3)-1
; CHECK-NEXT: BPF.JT.0.0:
; CHECK-NEXT:         .long   [[m1]]
; CHECK-NEXT:         .long   [[m2]]
; CHECK-NEXT:         .size   BPF.JT.0.0, 8

; CHECK-NEXT: [[m1:.*]] = (({{.*}}-.LBPF.JX.0.1)>>3)-1
; CHECK-NEXT: [[m2:.*]] = (({{.*}}-.LBPF.JX.0.1)>>3)-1
; CHECK-NEXT: [[m3:.*]] = (({{.*}}-.LBPF.JX.0.1)>>3)-1
; CHECK-NEXT: BPF.JT.0.1:
; CHECK-NEXT:         .long [[m1]]
; CHECK-NEXT:         .long [[m2]]
; CHECK-NEXT:         .long [[m3]]
; CHECK-NEXT:         .size BPF.JT.0.1, 12

entry:
  switch i64 %v1, label %sw.default [
    i64 0, label %sw.bb1
    i64 1, label %sw.bb2
  ]

sw.bb1:
  br label %sw.epilog

sw.bb2:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  %ret = phi i64 [ 42, %sw.default ], [ 3, %sw.bb1 ], [ 5, %sw.bb2 ]
  switch i64 %v2, label %sw.default.1 [
    i64 0, label %sw.bb1.1
    i64 1, label %sw.bb2.1
    i64 2, label %sw.bb3.1
  ]

sw.bb1.1:
  br label %sw.epilog.1

sw.bb2.1:
  br label %sw.epilog.1

sw.bb3.1:
  br label %sw.epilog.1

sw.default.1:
  br label %sw.epilog.1

sw.epilog.1:
  %ret.1 = phi i64 [ 42, %sw.default.1 ], [ 3, %sw.bb1.1 ], [ 5, %sw.bb2.1 ], [ 7, %sw.bb3.1 ]
  %ret.2 = add i64 %ret, %ret.1
  ret i64 %ret.2
}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang some version"}
