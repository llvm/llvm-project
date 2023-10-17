; RUN: llc -mtriple x86_64-pc-win32 < %s | FileCheck %s
; RUN: llc -mtriple x86_64-pc-win32 -jumptable-in-function-section < %s | FileCheck --check-prefixes=CHECK-OPT %s

define void @f(i32 %x) {
entry:
  switch i32 %x, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:
  tail call void @g(i32 0, i32 4)
  br label %sw.epilog

sw.bb1:
  tail call void @g(i32 1, i32 5)
  br label %sw.epilog

sw.bb2:
  tail call void @g(i32 2, i32 6)
  br label %sw.epilog

sw.bb3:
  tail call void @g(i32 3, i32 7)
  br label %sw.epilog

sw.epilog:
  tail call void @g(i32 10, i32 8)
  ret void
}

declare void @g(i32, i32)
; CHECK: .section        .rdata
; CHECK-OPT-NOT: .section        .rdata
