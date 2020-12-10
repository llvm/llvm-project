; RUN: llc -mtriple=x86_64-apple-unknown -stop-before=finalize-isel %s -o - -simplify-mir | FileCheck %s
; The byval argument of the function will be allocated a fixed stack slot. Test
; that we serialize the fixed slot correctly.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-unknown"

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

define hidden void @foo(i32* byval(i32) %dstRect) {
; CHECK-LABEL: name: foo
entry:
  call void @llvm.dbg.declare(metadata i32* %dstRect, metadata !3, metadata !DIExpression()), !dbg !5
; CHECK: fixedStack:
; CHECK: id: 0
; CHECK: debug-info-variable: '!3'
; CHECK: debug-info-expression: '!DIExpression()'
; CHECK: debug-info-location: '!5'
  unreachable
}

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "file.cpp", directory: "/dir")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(name: "dstRect", scope: !4)
!4 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !0, file: !1, line: 42, unit: !0)
!5 = !DILocation(line: 42, column: 85, scope: !4)
