; RUN: opt -passes=mergefunc -disable-output < %s
; This used to trigger a ConstantExpr::getBitCast assertion.

define void @t1() unnamed_addr uwtable ssp align 2 {
entry:
  switch i32 undef, label %sw.bb12 [
    i32 127, label %sw.bb
    i32 126, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry
  unreachable

sw.bb4:                                           ; preds = %entry
  unreachable

sw.bb12:                                          ; preds = %entry
  ret void
}

define void @t2() unnamed_addr uwtable ssp align 2 {
entry:
  switch i32 undef, label %sw.bb8 [
    i32 4, label %sw.bb
    i32 3, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry
  unreachable

sw.bb4:                                           ; preds = %entry
  ret void

sw.bb8:                                           ; preds = %entry
  unreachable
}
