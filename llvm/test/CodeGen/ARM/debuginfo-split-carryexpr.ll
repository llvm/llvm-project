; RUN: llc < %s -mtriple=armv7-linux-gnueabihf -stop-before=finalize-isel | FileCheck %s

; When splitting-up integers during CodeGen, if the debug info contains
; an expression with carry operators (i.e. arithmetic and shift ops), the
; debug information will be dropped as these operators cannot be correctly
; expressed across different registers.

; CHECK: [[HIGH:![0-9]+]] = !DILocalVariable(name: "high"
; CHECK: [[LOW:![0-9]+]] = !DILocalVariable(name: "low"
;
; As the debug information for "high" contains an arithmetic shift, while for
; "low" it does not, only the former should be undefined.
; CHECK-LABEL: body:
; CHECK: [[LOWR:%[0-9]+]]:gpr = COPY $r0
; CHECK: DBG_VALUE [[LOWR]], $noreg, [[LOW]]
; CHECK: DBG_VALUE $noreg, $noreg, [[HIGH]]
; CHECK: DBG_VALUE $noreg, $noreg, [[HIGH]]

define dso_local i64 @_Z2fnx(i64 returned %value) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i64 %value, metadata !13, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i64 %value, metadata !14, metadata !DIExpression(DW_OP_constu, 32, DW_OP_shra, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !17
  call void @llvm.dbg.value(metadata i64 %value, metadata !16, metadata !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !17
  ret i64 %value, !dbg !18
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0"}
!7 = distinct !DISubprogram(name: "fn", linkageName: "_Z2fnx", scope: !8, file: !8, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DIFile(filename: "test.cpp", directory: "/tmp")
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!12 = !{!13, !14, !16}
!13 = !DILocalVariable(name: "value", arg: 1, scope: !7, file: !8, line: 2, type: !11)
!14 = !DILocalVariable(name: "high", scope: !7, file: !8, line: 3, type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DILocalVariable(name: "low", scope: !7, file: !8, line: 4, type: !15)
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 5, column: 3, scope: !7)
