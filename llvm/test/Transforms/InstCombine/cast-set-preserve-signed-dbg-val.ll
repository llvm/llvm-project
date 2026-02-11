; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; CHECK-LABEL: define {{.*}} @test5
define i16 @test5(i16 %A) !dbg !34 {
  ; CHECK: [[and:%.*]] = and i16 %A, 15

  %B = sext i16 %A to i32, !dbg !40
  call void @llvm.dbg.value(metadata i32 %B, metadata !36, metadata !DIExpression()), !dbg !40

  %C = and i32 %B, 15, !dbg !41
  call void @llvm.dbg.value(metadata i32 %C, metadata !37, metadata !DIExpression()), !dbg !41

  ; Preserve the dbg.value for the DCE'd 32-bit 'and'.
  ; CHECK-NEXT: #dbg_value(i16 [[and]], [[C:![0-9]+]],
  ; CHECK-SAME:    !DIExpression(DW_OP_LLVM_convert, 16, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value)

  %D = trunc i32 %C to i16, !dbg !42
  call void @llvm.dbg.value(metadata i16 %D, metadata !38, metadata !DIExpression()), !dbg !42

  ; The dbg.value for a truncate should simply point to the result of the 16-bit 'and'.
  ; CHECK-NEXT: #dbg_value(i16 [[and]], [[D:![0-9]+]], !DIExpression(),

  ret i16 %D, !dbg !43
  ; CHECK-NEXT: ret i16 [[and]]
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "void", directory: "/")
!2 = !{}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !DISubroutineType(types: !2)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_signed)
!12 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_signed)
!34 = distinct !DISubprogram(name: "test5", linkageName: "test5", scope: null, file: !1, line: 12, type: !7, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !0, retainedNodes: !35)
!35 = !{!36, !37, !38}
!36 = !DILocalVariable(name: "B", scope: !34, file: !1, line: 12, type: !10)
!37 = !DILocalVariable(name: "C", scope: !34, file: !1, line: 13, type: !10)
!38 = !DILocalVariable(name: "D", scope: !34, file: !1, line: 14, type: !39)
!39 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_signed)
!40 = !DILocation(line: 12, column: 1, scope: !34)
!41 = !DILocation(line: 13, column: 1, scope: !34)
!42 = !DILocation(line: 14, column: 1, scope: !34)
!43 = !DILocation(line: 15, column: 1, scope: !34)
