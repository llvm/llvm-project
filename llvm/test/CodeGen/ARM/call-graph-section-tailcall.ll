;; Tests that we store the type identifiers in .llvm.callgraph section of the object file for tailcalls.

; RUN: llc -mtriple=arm-unknown-linux --call-graph-section -filetype=obj -o - < %s | \
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

!0 = !{!"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{!"_ZTSFicE.generalized"}
!3 = !{!"_ZTSFiiE.generalized"}

; CHECK: Hex dump of section '.llvm.callgraph':
; CHECK-NEXT: 0x00000000 000549d1 0e59ee5f f4520000 00008e19 ..I..Y._.R......
; CHECK-NEXT: 0x00000010 0b7f3326 e3000154 86bc5981 4b8e3000 ..3&...T..Y.K.0.
; CHECK-NEXT: 0x00000020 0549d10e 59ee5ff4 52000000 00a150b8 .I..Y._.R.....P.
; CHECK-NEXT: 0x00000030 3e0cfe3c b2015486 bc59814b 8e30     >..<..T..Y.K.0
