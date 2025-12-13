; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=1 | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=10 | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=100 | FileCheck %s

; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=1 -fast-isel=true | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=10 -fast-isel=true | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=100 -fast-isel=true | FileCheck %s

; This test has the same purpose as the instr-ref-opt-bisect.ll, to check if
; during opt-bisect's optimisation level change we won't run into an assert.
; This is simply testing different IR.

; CHECK: DBG_VALUE

target triple = "x86_64-pc-windows-msvc"

define i1 @foo(i32 %arg) !dbg !3 {
entry:
    #dbg_value(i32 %arg, !4, !DIExpression(), !5)
  switch i32 %arg, label %bb [
    i32 810, label %bb
  ], !dbg !5
bb:
  %a = load volatile i1, ptr null, align 1
  ret i1 false
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
!1 = !DIFile(filename: "instr-ref-opt-bisect2.ll", directory: ".")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "instr-ref-opt-bisect2", file: !1, unit: !0)
!4 = !DILocalVariable(name: "arg", arg: 2, scope: !3)
!5 = !DILocation(line: 0, scope: !3)
