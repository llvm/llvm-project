; RUN: llvm-as -disable-verify %s -o %t.bc
; ---- Full LTO ---------------------------------------------
; RUN: llc -filetype=asm -o - %t.bc 2>&1 | FileCheck %s
; CHECK-NOT: Broken module found
; CHECK: warning{{.*}} ignoring invalid debug info
; CHECK-NOT: Broken module found
; CHECK: foo
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12"

declare void @bar()

define void @foo() {
  call void @bar()
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "broken", directory: "")
