; RUN: llvm-as %s -o /dev/null

@foo = global i32 0
@bar = constant i32* getelementptr(i32* @foo)

