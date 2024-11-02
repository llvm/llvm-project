; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

@foo = global i32 0
@bar = constant ptr getelementptr(i32, ptr @foo)

