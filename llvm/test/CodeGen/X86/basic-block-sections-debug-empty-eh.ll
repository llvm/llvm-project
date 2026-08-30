; RUN: echo 'v1' > %t
; RUN: echo 'f f' >> %t
; RUN: echo 'c 0' >> %t
; RUN: echo 'c 1 2' >> %t
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu -function-sections \
; RUN:   -basic-block-sections=%t -verify-machineinstrs \
; RUN:   -stop-after=bbsections-prepare -o - | FileCheck %s

; A debug-only block at the start of a section must not suppress the NOP that
; prevents a zero-offset landing pad.
; CHECK:      bb.1.debug.only (bbsections 1, bb_id 1):
; CHECK:        DBG_VALUE
; CHECK:      bb.2.lpad (landing-pad, bbsections 1, bb_id 2):
; CHECK:        NOOP
; CHECK-NEXT:   EH_LABEL

@typeinfo = external constant ptr

declare void @g()
declare i32 @personality(...)

define void @f() personality ptr @personality !dbg !4 {
entry:
  invoke void @g()
          to label %debug.only unwind label %lpad

debug.only:
  #dbg_value(i1 false, !6, !DIExpression(), !8)
  unreachable

lpad:
  %result = landingpad { ptr, i32 }
          catch ptr @typeinfo
  unreachable
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1,
                             producer: "test", isOptimized: false,
                             runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1,
                            type: !5, scopeLine: 1,
                            spFlags: DISPFlagDefinition, unit: !0,
                            retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !DILocalVariable(name: "ghost", scope: !4, file: !1, line: 2, type: !7)
!7 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!8 = !DILocation(line: 2, column: 1, scope: !4)
