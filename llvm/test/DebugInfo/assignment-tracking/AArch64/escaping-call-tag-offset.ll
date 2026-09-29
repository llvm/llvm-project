; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android24"

declare void @use(ptr)

;; Test: Escaping call with DW_OP_LLVM_tag_offset in address expression.
;;
;; The dbg_assigns use !DIExpression(DW_OP_LLVM_tag_offset, 128) as the
;; address expression. After the escaping call to @use, the reinstated
;; DBG_VALUE should include the tag offset in its expression.

; CHECK-LABEL: name: test_tag_offset_escaping
; CHECK:       bb.0.entry:
; CHECK:         DBG_VALUE %stack.0.x, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_tag_offset, 128, DW_OP_deref)
; CHECK:         STRWui
; CHECK:         DBG_VALUE $noreg, $noreg, !{{[0-9]+}}, !DIExpression()
; CHECK:         BL @use
; CHECK:         DBG_VALUE %stack.0.x, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_tag_offset, 128, DW_OP_deref)

define void @test_tag_offset_escaping(i32 %a) !dbg !7 {
entry:
  %x = alloca i32, align 4, !DIAssignID !20
    #dbg_assign(i1 poison, !11, !DIExpression(), !20, ptr %x, !DIExpression(DW_OP_LLVM_tag_offset, 128), !12)
  store i32 1, ptr %x, align 4, !DIAssignID !21
    #dbg_assign(i32 1, !11, !DIExpression(), !21, ptr %x, !DIExpression(DW_OP_LLVM_tag_offset, 128), !12)
    #dbg_value(i32 %a, !11, !DIExpression(), !12)
  call void @use(ptr %x)
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!6 = !DISubroutineType(types: !2)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!7 = distinct !DISubprogram(name: "test_tag_offset_escaping", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocalVariable(name: "x", scope: !7, file: !1, line: 2, type: !14)
!12 = !DILocation(line: 2, column: 1, scope: !7)
!13 = !DILocation(line: 5, column: 1, scope: !7)
!20 = distinct !DIAssignID()
!21 = distinct !DIAssignID()