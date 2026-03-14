;RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
@var = global i32 0
@a = external global [2 x [2 x i32]], align 4

define void @test_inline_constraint_S() {
; CHECK-LABEL: test_inline_constraint_S:
  call void asm sideeffect "adrp x0, $0", "S"(ptr @var)
  call void asm sideeffect "add x0, x0, :lo12:$0", "S"(ptr @var)
  call void asm sideeffect "// $0", "S"(ptr getelementptr inbounds ([2 x [2 x i32]], ptr @a, i64 0, i64 1, i64 1))
; CHECK: adrp x0, var
; CHECK: add x0, x0, :lo12:var
; CHECK: // a+12
  ret void
}
define i32 @test_inline_constraint_S_label(i1 %in) {
; CHECK-LABEL: test_inline_constraint_S_label:
  call void asm sideeffect "adr x0, $0", "S"(ptr blockaddress(@test_inline_constraint_S_label, %loc))
; CHECK: adr x0, .Ltmp{{[0-9]+}}
br i1 %in, label %loc, label %loc2
loc:
  ret i32 0
loc2:
  ret i32 42
}
define i32 @test_inline_constraint_S_label_tailmerged(i1 %in) {
; CHECK-LABEL: test_inline_constraint_S_label_tailmerged:
  call void asm sideeffect "adr x0, $0", "S"(ptr blockaddress(@test_inline_constraint_S_label_tailmerged, %loc))
; CHECK: adr x0, .Ltmp{{[0-9]+}}
br i1 %in, label %loc, label %loc2
loc:
  br label %common.ret
loc2:
  br label %common.ret
common.ret:
  %common.retval = phi i32 [ 0, %loc ], [ 42, %loc2 ]
  ret i32 %common.retval
}

define i32 @test_inline_constraint_S_label_tailmerged2(i1 %in) {
; CHECK-LABEL: test_inline_constraint_S_label_tailmerged2:
  call void asm sideeffect "adr x0, $0", "S"(ptr blockaddress(@test_inline_constraint_S_label_tailmerged2, %loc))
; CHECK: adr x0, .Ltmp{{[0-9]+}}
  br i1 %in, label %loc, label %loc2
common.ret:
  %common.retval = phi i32 [ 0, %loc ], [ 42, %loc2 ]
  ret i32 %common.retval
loc:
  br label %common.ret
loc2:
  br label %common.ret
}
