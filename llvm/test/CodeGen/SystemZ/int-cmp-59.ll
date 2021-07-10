; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -O3 -stop-before=finalize-isel \
; RUN:   | FileCheck %s
;
; Check that an i1 in memory used in a comparison is loaded correctly.

@bPtr = external dso_local local_unnamed_addr global i32*, align 8
@c = external hidden unnamed_addr global i1, align 4

define i64 @main() {
; CHECK-LABEL: bb.0.entry:
; CHECK: %1:addr64bit = LARL @c
; CHECK: %2:gr64bit = LLGC %1, 0, $noreg :: (dereferenceable load (s8) from @c, align 4)
; CHECK-NEXT: %4:gr64bit = IMPLICIT_DEF
; CHECK-NEXT: %3:gr64bit = RISBGN %4, killed %2, 63, 191, 0
; CHECK-NEXT: %5:gr64bit = LCGR killed %3, implicit-def dead $cc
; CHECK-NEXT: CGHI killed %5, 1, implicit-def $cc
entry:
  %0 = load i32*, i32** @bPtr
  store i1 true, i1* @c
  store i32 8, i32* %0
  %.b = load i1, i1* @c
  %conv.i = select i1 %.b, i64 1, i64 3
  %div.i = sdiv i64 -1, %conv.i
  %cmp.i = icmp eq i64 %div.i, 1
  %conv2.i = zext i1 %cmp.i to i64
  ret i64 %conv2.i
}

