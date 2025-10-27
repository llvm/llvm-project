;; Tests that we store the type identifiers in .llvm.callgraph section of the object file for tailcalls.

; RUN: llc -mtriple=arm-unknown-linux --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .llvm.callgraph - | FileCheck %s

define i32 @check_tailcall(ptr %func, i8 %x) !type !0 {
entry:
  %call = tail call i32 %func(i8 signext %x), !callee_type !1
  ret i32 %call
}

define i32 @main(i32 %argc) !type !3 {
entry:
  %andop = and i32 %argc, 1
  %cmp = icmp eq i32 %andop, 0
  %foo.bar = select i1 %cmp, ptr @foo, ptr @bar
  %call.i = tail call i32 %foo.bar(i8 signext 97), !callee_type !1
  ret i32 %call.i
}

declare !type !2 i32 @foo(i8 signext)

declare !type !2 i32 @bar(i8 signext)

!0 = !{i64 0, !"_ZTSFiPvcE.generalized"}
!1 = !{!2}
!2 = !{i64 0, !"_ZTSFicE.generalized"}
!3 = !{i64 0, !"_ZTSFiiE.generalized"}

; CHECK:      Hex dump of section '.llvm.callgraph':
; CHECK-NEXT: 0x00000000 00050000 00008e19 0b7f3326 e3000154
; CHECK-NEXT: 0x00000010 86bc5981 4b8e3000 05100000 00a150b8
;; Verify that the type id 0x308e4b8159bc8654 is in section.
; CHECK-NEXT: 0x00000020 3e0cfe3c b2015486 bc59814b 8e30
