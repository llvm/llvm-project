; RUN: opt -passes="loop-mssa(simple-loop-unswitch)" -verify-memoryssa -disable-output < %s
; PR10031

define i32 @test(i32 %command, i1 %arg) {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then14, %tailrecurse, %entry
  br i1 %arg, label %if.then, label %tailrecurse

if.then:                                          ; preds = %tailrecurse
  switch i32 %command, label %sw.bb [
    i32 2, label %land.lhs.true
    i32 0, label %land.lhs.true
  ]

land.lhs.true:                                    ; preds = %if.then, %if.then
  br i1 %arg, label %sw.bb, label %if.then14

if.then14:                                        ; preds = %land.lhs.true
  switch i32 %command, label %tailrecurse [
    i32 0, label %sw.bb
    i32 1, label %sw.bb
  ]

sw.bb:                                            ; preds = %if.then14
  unreachable
}
