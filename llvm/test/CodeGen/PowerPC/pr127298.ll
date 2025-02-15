; RUN: llc < %s -mtriple=powerpc | FileCheck %s

define void @foo() {
Entry:
  %0 = load volatile i8, ptr inttoptr (i32 -559038737 to ptr), align 1
  ret void
}
