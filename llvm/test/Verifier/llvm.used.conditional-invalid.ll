; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

@a = global i32 42
@llvm.used = appending global [1 x i32*] [i32* @a], section "llvm.metadata"

@cond = global i32 43
!1 = !{ i32* @cond, i32 1, !{ i32* @a }, i32 8 }
!llvm.used.conditional = !{ !1 }

; CHECK: invalid llvm.used.conditional member
