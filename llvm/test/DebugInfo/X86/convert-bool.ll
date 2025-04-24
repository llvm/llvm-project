; RUN: llc -mtriple=x86_64 -dwarf-version=5 -filetype=obj -O0 < %s | llvm-dwarfdump - | FileCheck %s

; CHECK: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name      ("DW_ATE_unsigned_1")
; CHECK-NEXT: DW_AT_encoding  (DW_ATE_unsigned)
; CHECK-NEXT: DW_AT_byte_size (0x01)
; CHECK-NEXT: DW_AT_bit_size  (0x01)

define void @main() !dbg !18 {
entry:
    #dbg_value(i1 false, !22, !DIExpression(DW_OP_not, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value), !23)
  ret void, !dbg !24
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/", checksumkind: CSK_MD5, checksum: "100bdbce655d729c24c7c0e8523a58ae")
!2 = !{!3, !6, !8}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(name: "a", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "b", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "c", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!17 = !{!"clang version 21.0.0git"}
!18 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !19, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !{!22}
!22 = !DILocalVariable(name: "l_4516", scope: !18, file: !1, line: 4, type: !5)
!23 = !DILocation(line: 0, scope: !18)
!24 = !DILocation(line: 9, column: 1, scope: !18)
