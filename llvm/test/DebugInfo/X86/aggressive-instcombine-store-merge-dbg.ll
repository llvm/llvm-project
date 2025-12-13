; RUN: opt -S -passes=aggressive-instcombine -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

;; Aggressive instcombine merges the two i8 stores into an i16 store. Check
;; the debug location and DIAssignID metadata get merged.

; CHECK: define void @test_i16(i16 %x, ptr %p) !dbg ![[#]] {
; CHECK-NEXT: store i16 %x, ptr %p, align 1, !dbg ![[DBG:[0-9]+]], !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i16 %x, ![[#]],
;    CHECK-SAME: !DIExpression(DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 8),
;    CHECK-SAME: ![[ID]], ptr %p, !DIExpression(), ![[#]])
; CHECK-NEXT: #dbg_assign(i16 %x, ![[#]],
;    CHECK-SAME: !DIExpression(DW_OP_constu, 8, DW_OP_shr, DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value, DW_OP_LLVM_fragment, 8, 8),
;    CHECK-SAME: ![[ID]], ptr %p, !DIExpression(DW_OP_plus_uconst, 1), ![[#]])
; CHECK-NEXT: ret void

; CHECK: ![[DBG]] = !DILocation(line: 0, scope: ![[#]])

define void @test_i16(i16 %x, ptr %p) !dbg !5 {
  %x.0 = trunc i16 %x to i8
  store i8 %x.0, ptr %p, align 1, !dbg !16, !DIAssignID !17
    #dbg_assign(i8 %x.0, !9, !DIExpression(DW_OP_LLVM_fragment, 0, 8), !17, ptr %p, !DIExpression(), !18)
  %shr.1 = lshr i16 %x, 8
  %x.1 = trunc i16 %shr.1 to i8
  %gep.1 = getelementptr i8, ptr %p, i64 1
  store i8 %x.1, ptr %gep.1, align 1, !dbg !19, !DIAssignID !20
    #dbg_assign(i8 %x.1, !9, !DIExpression(DW_OP_LLVM_fragment, 8, 8), !20, ptr %gep.1, !DIExpression(), !18)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/app/example.ll", directory: "/")
!2 = !{i32 7}
!3 = !{i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_i16", linkageName: "test_i16", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
!16 = !DILocation(line: 2, column: 1, scope: !5)
!17 = distinct !DIAssignID()
!18 = !DILocation(line: 1, column: 1, scope: !5)
!19 = !DILocation(line: 6, column: 1, scope: !5)
!20 = distinct !DIAssignID()
