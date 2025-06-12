; RUN: llvm-module-splitter %s | FileCheck %s

; CHECK-LABEL: [LLVM Module Split: submodule 0]

; CHECK: define void @foo
define void @foo() {
  call void @baz()
  ret void
}

; CHECK: define void @baz
define void @baz() {
  ret void
}

; CHECK: define void @bar
define void @bar() {
  call void @baz()
  ret void
}

; CHECK-LABEL: [LLVM Module Split: submodule 1]

; CHECK: define void @boo
define void @boo() {
  ret void
}
