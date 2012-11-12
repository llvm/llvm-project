; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s

@zed = external hidden constant i32

define hidden void @foo() nounwind {
entry:
  call void @bar(i32* @zed)
  ret void
}

declare hidden void @bar(i32*)

;CHECK: .hidden	zed
;CHECK: .hidden	bar
