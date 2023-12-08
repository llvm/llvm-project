; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s

@zed = external hidden constant i32

define available_externally hidden void @baz() {
  ret void
}

define hidden void @foo() nounwind {
entry:
  call void @bar(ptr @zed)
  call void @baz()
  ret void
}

declare hidden void @bar(ptr)

;CHECK: .hidden	zed
;CHECK: .hidden	baz
;CHECK: .hidden	bar
