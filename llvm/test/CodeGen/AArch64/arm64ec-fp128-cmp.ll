; GlobalISel is the default selector at -O0, but Arm64EC call lowering is
; unimplemented there, so fp128 compares must fall back to SelectionDAG rather
; than silently dropping the libcall.
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -O0 < %s | FileCheck %s
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -O2 < %s | FileCheck %s

define i1 @cmp_oeq(fp128 %a, fp128 %b) {
; CHECK-LABEL: "#cmp_oeq":
; CHECK: bl "#__eqtf2"
  %r = fcmp oeq fp128 %a, %b
  ret i1 %r
}

define i1 @cmp_olt(fp128 %a, fp128 %b) {
; CHECK-LABEL: "#cmp_olt":
; CHECK: bl "#__lttf2"
  %r = fcmp olt fp128 %a, %b
  ret i1 %r
}

define i1 @cmp_uno(fp128 %a, fp128 %b) {
; CHECK-LABEL: "#cmp_uno":
; CHECK: bl "#__unordtf2"
  %r = fcmp uno fp128 %a, %b
  ret i1 %r
}

define i1 @cmp_une_self(fp128 %a) {
; CHECK-LABEL: "#cmp_une_self":
; CHECK: bl "#__unordtf2"
  %r = fcmp une fp128 %a, %a
  ret i1 %r
}
