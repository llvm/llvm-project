; RUN: opt -S -passes=licm,indvars -verify-memoryssa < %s | FileCheck %s
; REQUIRES: asserts
@v_69 = external constant { i16, i16 }, align 1

; CHECK-LABEL: @f()
define void @f() {
entry:
  br label %for.cond26

for.cond26:                                       ; preds = %for.body28, %entry
  br i1 true, label %for.body28, label %for.cond.cleanup27

for.cond.cleanup27:                               ; preds = %for.cond26
  unreachable

for.body28:                                       ; preds = %for.cond26
  %v_69.imag = load volatile i16, i16* getelementptr inbounds ({ i16, i16 }, { i16, i16 }* @v_69, i32 0, i32 1), align 1
  %.real42 = load i32, i32* undef, align 1
  store i32 %.real42, i32* undef, align 1
  br label %for.cond26
}
