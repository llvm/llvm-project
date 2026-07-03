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

!0 = !{!"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{!"_ZTSFicE.generalized"}
!3 = !{!"_ZTSFiiE.generalized"}

; CHECK: Hex dump of section '.llvm.callgraph':
; CHECK-NEXT: 0x00000000 000549d1 0e59ee5f f4520000 00000000 ..I..Y._.R......
; CHECK-NEXT: 0x00000010 00008e19 0b7f3326 e3000154 86bc5981 ......3&...T..Y.
; CHECK-NEXT: 0x00000020 4b8e3000 0549d10e 59ee5ff4 52000000 K.0..I..Y._.R...
; CHECK-NEXT: 0x00000030 00000000 00a150b8 3e0cfe3c b2015486 ......P.>..<..T.
; CHECK-NEXT: 0x00000040 bc59814b 8e30                       .Y.K.0
