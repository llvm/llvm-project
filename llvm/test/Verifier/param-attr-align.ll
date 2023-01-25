; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Attribute 'align' exceed the max size 2^14
define dso_local void @foo(ptr %p) {
entry:
  call void @bar(ptr noundef byval(<8 x float>) align 32768 %p)
  ret void
}

declare dso_local void @bar(ptr %p)
