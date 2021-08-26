; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

@A = global i1 0, align 1073741824

define void @foo() {
  %p = alloca i1, align 1073741824
  load i1, i1* %p, align 1073741824
  store i1 false, i1* %p, align 1073741824
  ret void
}
