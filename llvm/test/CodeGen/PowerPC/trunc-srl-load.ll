; RUN: llc -mtriple=powerpc64-unknown-unknown %s -o - | FileCheck %s

; CHECK-LABEL: trunc_srl_load
; CHECK-NOT: lhz 4, 4(0)
; CHECK: lhz 4, 2(0)
define dso_local fastcc void @trunc_srl_load(i32 zeroext %AttrArgNo) {
entry:
  %bf.load.i = load i64, i64* null, align 8
  %bf.lshr.i = lshr i64 %bf.load.i, 32
  %0 = trunc i64 %bf.lshr.i to i32
  %bf.cast.i = and i32 %0, 65535
  %cmp.i = icmp ugt i32 %bf.cast.i, %AttrArgNo
  br i1 %cmp.i, label %exit, label %cond.false
exit:       ; preds = %entry
  unreachable
cond.false:                                       ; preds = %entry
  unreachable
}
