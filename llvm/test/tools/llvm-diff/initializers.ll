; RUN: llvm-diff %s %s

; An initializer that has a GEP instruction in it won't match itself in
; llvm-diff unless the a deep comparison is done on the initializer.

@gv1 = external dso_local global [28 x i16], align 16
@gv2 = private unnamed_addr constant [2 x i16*] [i16* getelementptr inbounds ([28 x i16], [28 x i16]* @gv1, i32 0, i32 0), i16* poison], align 16

define void @foo() {
  %1 = getelementptr [2 x i16*], [2 x i16*]* @gv2, i64 0, i64 undef
  ret void
}
