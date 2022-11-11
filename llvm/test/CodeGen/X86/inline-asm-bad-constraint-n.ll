; RUN: not llc -mtriple=i686-- -no-integrated-as < %s 2>&1 | FileCheck %s

@x = global i32 0, align 4

; CHECK: error: constraint 'n' expects an integer constant expression
define void @foo() {
  %a = getelementptr i32, ptr @x, i32 1
  call void asm sideeffect "foo $0", "n"(ptr %a) nounwind
  ret void
}
