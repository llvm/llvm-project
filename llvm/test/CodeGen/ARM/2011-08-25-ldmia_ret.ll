; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a9 | FileCheck %s
; Test that ldmia_ret preserves implicit operands for return values.
;
; This CFG is reduced from a benchmark miscompile. With current
; if-conversion heuristics, one of the return paths is if-converted
; into sw.bb18 resulting in an ldmia_ret in the middle of the
; block. The postra scheduler needs to know that the return implicitly
; uses the return register, otherwise its antidep breaker scavenges
; the register in order to hoist the constant load required to test
; the switch.

declare i32 @getint()
declare i1 @getbool()
declare void @foo(i32)
declare i32 @bar(i32)

define i32 @test(i32 %in1, i32 %in2) nounwind {
entry:
  %call = tail call zeroext i1 @getbool() nounwind
  br i1 %call, label %sw.bb18, label %sw.bb2

sw.bb2:                                           ; preds = %entry
  %cmp = tail call zeroext i1 @getbool() nounwind
  br i1 %cmp, label %sw.epilog58, label %land.lhs.true

land.lhs.true:                                    ; preds = %sw.bb2
  %cmp13 = tail call zeroext i1 @getbool() nounwind
  br i1 %cmp13, label %if.then, label %sw.epilog58

if.then:                                          ; preds = %land.lhs.true
  tail call void @foo(i32 %in1) nounwind
  br label %sw.epilog58

; load the return value
; CHECK: movs	[[RRET:r.]], #2
; hoist the switch constant without clobbering RRET
; CHECK: movw
; CHECK-NOT: [[RRET]]
; CHECK: , #63707
; CHECK-NOT: [[RRET]]
; CHECK: tst
; If-convert the return
; CHECK: it	ne
; Fold the CSR+return into a pop
; CHECK: popne	{r4, r5, r7, pc}
sw.bb18:
  %call20 = tail call i32 @bar(i32 %in2) nounwind
  switch i32 %call20, label %sw.default56 [
    i32 168, label %sw.bb21
    i32 165, label %sw.bb21
    i32 261, label %sw.epilog58
    i32 188, label %sw.epilog58
    i32 187, label %sw.epilog58
    i32 186, label %sw.epilog58
    i32 185, label %sw.epilog58
    i32 184, label %sw.epilog58
    i32 175, label %sw.epilog58
    i32 174, label %sw.epilog58
    i32 173, label %sw.epilog58
    i32 172, label %sw.epilog58
    i32 171, label %sw.epilog58
    i32 167, label %sw.epilog58
    i32 166, label %sw.epilog58
    i32 164, label %sw.epilog58
    i32 163, label %sw.epilog58
    i32 161, label %sw.epilog58
    i32 160, label %sw.epilog58
    i32 -1, label %sw.bb33
  ]

sw.bb21:                                          ; preds = %sw.bb18, %sw.bb18
  tail call void @foo(i32 %in2) nounwind
  %call28 = tail call i32 @getint() nounwind
  %tobool = icmp eq i32 %call28, 0
  br i1 %tobool, label %if.then29, label %sw.epilog58

if.then29:                                        ; preds = %sw.bb21
  tail call void @foo(i32 %in2) nounwind
  br label %sw.epilog58

sw.bb33:                                          ; preds = %sw.bb18
  %cmp42 = tail call zeroext i1 @getbool() nounwind
  br i1 %cmp42, label %sw.default56, label %land.lhs.true44

land.lhs.true44:                                  ; preds = %sw.bb33
  %call50 = tail call i32 @getint() nounwind
  %cmp51 = icmp slt i32 %call50, 0
  br i1 %cmp51, label %if.then53, label %sw.default56

if.then53:                                        ; preds = %land.lhs.true44
  tail call void @foo(i32 %in2) nounwind
  br label %sw.default56

sw.default56:                                     ; preds = %sw.bb33, %land.lhs.true44, %if.then53, %sw.bb18
  br label %sw.epilog58

sw.epilog58:
  %retval.0 = phi i32 [ 4, %sw.default56 ], [ 2, %sw.bb21 ], [ 2, %if.then29 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb18 ], [ 2, %sw.bb2 ], [ 2, %land.lhs.true ], [ 2, %if.then ]
  ret i32 %retval.0
}
