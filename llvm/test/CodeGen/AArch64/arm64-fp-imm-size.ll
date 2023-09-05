; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-darwin -global-isel | FileCheck %s

; CHECK: literal8
; CHECK: .quad  0x400921fb54442d18
define double @foo() optsize {
; CHECK: _foo:
; CHECK: adrp x[[REG:[0-9]+]], lCPI0_0@PAGE
; CHECK: ldr  d0, [x[[REG]], lCPI0_0@PAGEOFF]
; CHECK-NEXT: ret
  ret double 0x400921FB54442D18
}

; CHECK: literal8
; CHECK: .quad 0x0000001fffffffc
define double @foo2() optsize {
; CHECK: _foo2:
; CHECK: adrp x[[REG:[0-9]+]], lCPI1_0@PAGE
; CHECK: ldr  d0, [x[[REG]], lCPI1_0@PAGEOFF]
; CHECK-NEXT: ret
  ret double 0x1FFFFFFFC1
}

define float @bar() optsize {
; CHECK: _bar:
; CHECK: adrp x[[REG:[0-9]+]], lCPI2_0@PAGE
; CHECK: ldr  s0, [x[[REG]], lCPI2_0@PAGEOFF]
; CHECK-NEXT:  ret
  ret float 0x400921FB60000000
}

; CHECK: literal16
; CHECK: .quad 0
; CHECK: .quad 0
define fp128 @baz() optsize {
; CHECK: _baz:
; CHECK:  adrp x[[REG:[0-9]+]], lCPI3_0@PAGE
; CHECK:  ldr  q0, [x[[REG]], lCPI3_0@PAGEOFF]
; CHECK-NEXT:  ret
  ret fp128 0xL00000000000000000000000000000000
}

; CHECK: literal8
; CHECK: .quad 0x0000001fffffffd
define double @foo2_pgso() !prof !14 {
; CHECK: _foo2_pgso:
; CHECK: adrp x[[REG:[0-9]+]], lCPI4_0@PAGE
; CHECK: ldr  d0, [x[[REG]], lCPI4_0@PAGEOFF]
; CHECK-NEXT: ret
  ret double 0x1FFFFFFFd1
}

define float @bar_pgso() !prof !14 {
; CHECK: _bar_pgso:
; CHECK: adrp x[[REG:[0-9]+]], lCPI5_0@PAGE
; CHECK: ldr  s0, [x[[REG]], lCPI5_0@PAGEOFF]
; CHECK-NEXT:  ret
  ret float 0x400921FB80000000
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
