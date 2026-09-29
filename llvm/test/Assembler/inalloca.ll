; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

define void @a() {
entry:
  %0 = alloca inalloca i32
  %1 = alloca inalloca [2 x i32]
  %2 = alloca inalloca i32, i32 2
  %3 = alloca inalloca i32, i32 2, align 16
  %4 = alloca inalloca i32, i32 2, align 16, !foo !0
  %5 = alloca i32, i32 2, align 16, !foo !0
  %6 = alloca i32, i32 2, align 16
  ret void
}

!0 = !{i32 662302, null}
!foo = !{ !0 }
