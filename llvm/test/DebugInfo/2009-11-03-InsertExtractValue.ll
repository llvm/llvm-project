; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!0 = metadata !{i32 42}

define <{i32, i32}> @f1() {
; CHECK: !dbgx !0
  %r = insertvalue <{ i32, i32 }> zeroinitializer, i32 4, 1, !dbgx !0
; CHECK: !dbgx !0
  %e = extractvalue <{ i32, i32 }> %r, 0, !dbgx !0
  ret <{ i32, i32 }> %r
}
