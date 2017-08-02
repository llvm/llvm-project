; RUN: not llc -mtriple=i686-- -no-integrated-as < %s 2>&1 | FileCheck %s

@x = global i32 0, align 4

;CHECK:	error: invalid operand for inline asm constraint 'n'
define void @foo() {
  %a = getelementptr i32, i32* @x, i32 1
  call void asm sideeffect "foo $0", "n"(i32* %a) nounwind
  ret void
}
