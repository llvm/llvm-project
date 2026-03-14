; RUN: not opt --mtriple x86_64-unknown-linux-gnu < %s -passes=embed-bitcode -S 2>&1 | FileCheck %s

@a = global i32 1
@llvm.embedded.module = private constant [4 x i8] c"BC\C0\DE"

; CHECK: LLVM ERROR: Can only embed the module once
