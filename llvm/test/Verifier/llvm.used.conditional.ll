; RUN: llvm-as < %s -o /dev/null

@a = global i32 42
@llvm.used = appending global [1 x i32*] [i32* @a], section "llvm.metadata"

@cond = global i32 43
!1 = !{ i32* @cond, i32 1, !{ i32* @a } }
!2 = !{ null,       i32 1, !{ i32* @a } }
!llvm.used.conditional = !{ !1, !2 }
