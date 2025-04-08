; RUN: opt -S -passes=structurizecfg %s -o -
; REQUIRES: asserts
; XFAIL: *

; Issue tracking: https://github.com/llvm/llvm-project/issues/126534.
; FIXME: This test is expected to crash. Generate checklines after the crash is fixed.

define void @foo() {
entry:
  br i1 false, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  ret void
}
