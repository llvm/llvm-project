; RUN: llc < %s -mtriple=arm64-eabi

; The target lowering for integer comparisons was replacing some DAG nodes
; during operation legalization, which resulted in dangling pointers,
; cycles in DAGs, and eventually crashes.  This is the testcase for
; one of those crashes. (rdar://10653656)

define void @test(i1 zeroext %IsArrow) nounwind ssp align 2 {
entry:
  br i1 undef, label %return, label %lor.lhs.false

lor.lhs.false:
  br i1 undef, label %return, label %if.end

if.end:
  %tmp.i = load i64, i64* undef, align 8
  %and.i.i.i = and i64 %tmp.i, -16
  br i1 %IsArrow, label %if.else_crit_edge, label %if.end32

if.else_crit_edge:
  br i1 undef, label %if.end32, label %return

if.end32:
  %0 = icmp ult i32 undef, 3
  %1 = zext i64 %tmp.i to i320
  %.pn.v = select i1 %0, i320 128, i320 64
  %.pn = shl i320 %1, %.pn.v
  %ins346392 = or i320 %.pn, 0
  store i320 %ins346392, i320* undef, align 8
  br i1 undef, label %sw.bb.i.i, label %exit

sw.bb.i.i:
  unreachable

exit:
  unreachable

return:
  ret void
}
