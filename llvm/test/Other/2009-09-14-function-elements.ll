; RUN: not llvm-as %s -disable-output 2>/dev/null

; Arrays and structures with function types (not function pointers) are illegal.

@foo = external global [4 x i32 (i32)]
@bar = external global { i32 (i32) }
