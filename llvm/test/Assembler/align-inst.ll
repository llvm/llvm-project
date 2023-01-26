; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

@A = global i1 0, align 4294967296

define void @foo() {
  %p = alloca i1, align 4294967296
  load i1, ptr %p, align 4294967296
  store i1 false, ptr %p, align 4294967296
  ret void
}
