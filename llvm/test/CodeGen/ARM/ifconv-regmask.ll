; RUN: llc < %s -mtriple=thumbv7s-apple-ios6.0.0 -verify-machineinstrs

%union.opcode = type { i32 }

@opcode = external global %union.opcode, align 4

; Function Attrs: nounwind ssp
define i32 @sfu() {
entry:
  %bf.load = load i32, ptr @opcode, align 4
  %bf.lshr = lshr i32 %bf.load, 26
  %bf.clear = and i32 %bf.lshr, 7
  switch i32 %bf.clear, label %return [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
  ]

sw.bb:                                            ; preds = %entry
  %call = tail call i32 @func0()
  br label %return

sw.bb1:                                           ; preds = %entry
  %call2 = tail call i32 @func1()
  br label %return

return:                                           ; preds = %sw.bb1, %sw.bb, %entry
  %retval.0 = phi i32 [ %call2, %sw.bb1 ], [ %call, %sw.bb ], [ -1, %entry ]
  ret i32 %retval.0
}

; Function Attrs: nounwind ssp
declare i32 @func0()

; Function Attrs: nounwind ssp
declare i32 @func1()
