; RUN: not llc -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
; CHECK:      error: couldn't allocate input reg for constraint '{d2}'
; CHECK-NEXT: error: couldn't allocate input reg for constraint '{s2}'
; CHECK-NEXT: error: couldn't allocate input reg for constraint '{d3}'

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8a-unknown-linux-gnueabihf"

@a = local_unnamed_addr global i32 0, align 4

define void @_Z1bv() local_unnamed_addr {
entry:
  %0 = load i32, ptr @a, align 4
  %conv = sext i32 %0 to i64
  tail call void asm sideeffect "", "{d2}"(i64 %conv)
  ret void
}

define void @_Z1cv() local_unnamed_addr {
entry:
  %0 = load i32, ptr @a, align 4
  %conv = sext i32 %0 to i64
  tail call void asm sideeffect "", "{s2}"(i64 %conv)
  ret void
}

define void @_Z1dv() local_unnamed_addr {
entry:
  tail call void asm sideeffect "", "{d3}"(<16 x i8> splat (i8 -1))
  ret void
}
