; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: expected index

define void @f1() {
  extractvalue <{ i32, i32 }> undef, !dbg !0
  ret void
}
