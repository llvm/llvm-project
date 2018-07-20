;RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
@var = global i32 0
define void @test_inline_constraint_S() {
; CHECK-LABEL: test_inline_constraint_S:
  call void asm sideeffect "adrp x0, $0", "S"(i32* @var)
  call void asm sideeffect "add x0, x0, :lo12:$0", "S"(i32* @var)
; CHECK: adrp x0, var
; CHECK: add x0, x0, :lo12:var
  ret void
}
define i32 @test_inline_constraint_S_label(i1 %in) {
; CHECK-LABEL: test_inline_constraint_S_label:
  call void asm sideeffect "adr x0, $0", "S"(i8* blockaddress(@test_inline_constraint_S_label, %loc))
; CHECK: adr x0, .Ltmp{{[0-9]+}}
br i1 %in, label %loc, label %loc2
loc:
  ret i32 0
loc2:
  ret i32 42
}
