; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; many-dom.ll, with an added back-edge back into the switch.
; Because people love their gotos.

declare i1 @getBool() readnone

define i32 @foo(i32* %p) {
entry:
  br label %loopbegin

loopbegin:
; CHECK: 9 = MemoryPhi({entry,liveOnEntry},{sw.epilog,6})
; CHECK-NEXT: %n =
  %n = phi i32 [ 0, %entry ], [ %1, %sw.epilog ]
  %m = alloca i32, align 4
  switch i32 %n, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:
; CHECK: 1 = MemoryDef(9)
; CHECK-NEXT: store i32 1
  store i32 1, i32* %m, align 4
  br label %sw.epilog

sw.bb1:
; CHECK: 2 = MemoryDef(9)
; CHECK-NEXT: store i32 2
  store i32 2, i32* %m, align 4
  br label %sw.epilog

sw.bb2:
; CHECK: 3 = MemoryDef(9)
; CHECK-NEXT: store i32 3
  store i32 3, i32* %m, align 4
  br label %sw.epilog

sw.bb3:
; CHECK: 10 = MemoryPhi({loopbegin,9},{sw.almostexit,6})
; CHECK: 4 = MemoryDef(10)
; CHECK-NEXT: store i32 4
  store i32 4, i32* %m, align 4
  br label %sw.epilog

sw.default:
; CHECK: 5 = MemoryDef(9)
; CHECK-NEXT: store i32 5
  store i32 5, i32* %m, align 4
  br label %sw.epilog

sw.epilog:
; CHECK: 8 = MemoryPhi({sw.default,5},{sw.bb3,4},{sw.bb,1},{sw.bb1,2},{sw.bb2,3})
; CHECK-NEXT: MemoryUse(8)
; CHECK-NEXT: %0 =
  %0 = load i32, i32* %m, align 4
; CHECK: 6 = MemoryDef(8)
; CHECK-NEXT: %1 =
  %1 = load volatile i32, i32* %p, align 4
  %2 = icmp eq i32 %0, %1
  br i1 %2, label %sw.almostexit, label %loopbegin

sw.almostexit:
  %3 = icmp eq i32 0, %1
  br i1 %3, label %exit, label %sw.bb3

exit:
; CHECK: 7 = MemoryDef(6)
; CHECK-NEXT: %4 = load volatile i32
  %4 = load volatile i32, i32* %p, align 4
  %5 = add i32 %4, %1
  ret i32 %5
}
