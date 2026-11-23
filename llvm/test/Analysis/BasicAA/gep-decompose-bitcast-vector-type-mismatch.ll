; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck  %s

; Verify that BasicAA does not crash when DecomposeGEPExpression encounters
; a bitcast whose stripped operand has a non-pointer type (e.g. <1 x ptr>).
; The alias query should conservatively return MayAlias instead of hitting
; an assertion in alias() which requires scalar pointer types.
define void @test(ptr %C, i64 %0) {
; CHECK-LABEL: test
; CHECK: MayAlias:    i32* %2, <2 x i16>* bitcast (<1 x ptr> <ptr inttoptr (i64 -1 to ptr)> to ptr)
  %2 = getelementptr i8, ptr %C, i64 %0
  store i32 0, ptr %2, align 4
  store <2 x i16> zeroinitializer, ptr bitcast (<1 x ptr> <ptr inttoptr (i64 -1 to ptr)> to ptr), align 4
  ret void
}
