;; Tests that we store the type identifiers in .llvm.callgraph section of the object file for tailcalls.

; REQUIRES: x86-registered-target
; REQUIRES: arm-registered-target

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .llvm.callgraph - | FileCheck %s

define i32 @check_tailcall(ptr %func, i8 %x) !callgraph !0 {
entry:
  %call = tail call i32 %func(i8 signext %x), !callee_type !1
  ret i32 %call
}

define i32 @main(i32 %argc) !callgraph !3 {
entry:
  %andop = and i32 %argc, 1
  %cmp = icmp eq i32 %andop, 0
  %foo.bar = select i1 %cmp, ptr @foo, ptr @bar
  %call.i = tail call i32 %foo.bar(i8 signext 97), !callee_type !1
  ret i32 %call.i
}

declare !callgraph !2 i32 @foo(i8 signext)

declare !callgraph !2 i32 @bar(i8 signext)

!0 = !{!"_ZTSFiPvcE"}
!1 = !{!2}
!2 = !{!"_ZTSFicE"}
!3 = !{!"_ZTSFiiE"}

; CHECK: Hex dump of section '.llvm.callgraph':
; CHECK-NEXT: 0x00000000 00050000 00000000 0000d4bf 88b60134
; CHECK-NEXT: 0x00000010 63f001c6 50144734 74f90100 05000000
;; Verify that the type id 0x144734744701c650 is in section.
; CHECK-NEXT: 0x00000020 00000000 00423a34 855a01ce 4701c650
; CHECK-NEXT: 0x00000030 14473474 f901
