; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s

define void @Main__Test() #0 !dbg !17 {
entry:
  %as = alloca i64, align 8
  %bs = alloca i64, align 8
  br label %second, !dbg !21

second:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata ptr %as, metadata !22, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata ptr %bs, metadata !26, metadata !DIExpression()), !dbg !25
  store i64 36028797018972298, ptr %as, align 8, !dbg !28
  store i64 85, ptr %bs, align 8, !dbg !29
  ret void, !dbg !21
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

!llvm.ident = !{!0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!14, !15, !16}

!0 = !{!"versions- cm3: d5.10.0 llvm: 12.0"}
!1 = distinct !DICompileUnit(language: DW_LANG_Modula3, file: !2, producer: "cm3", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3)
!2 = !DIFile(filename: "Main.m3", directory: "/home/peter/cm3/settest/src")
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum", scope: !2, file: !2, line: 11, size: 8, align: 8, elements: !5)
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13}
!6 = !DIEnumerator(name: "alpha", value: 0)
!7 = !DIEnumerator(name: "beta", value: 1)
!8 = !DIEnumerator(name: "gamma", value: 2)
!9 = !DIEnumerator(name: "delta", value: 3)
!10 = !DIEnumerator(name: "epsilon", value: 4)
!11 = !DIEnumerator(name: "theta", value: 5)
!12 = !DIEnumerator(name: "psi", value: 6)
!13 = !DIEnumerator(name: "zeta", value: 7)
!14 = !{i64 2, !"Dwarf Version", i64 4}
!15 = !{i64 2, !"Debug Info Version", i64 3}
!16 = !{i64 2, !"wchar_size", i64 2}
!17 = distinct !DISubprogram(name: "Test", linkageName: "Main__Test", scope: !2, file: !2, line: 11, type: !18, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{}
!21 = !DILocation(line: 20, scope: !17)
!22 = !DILocalVariable(name: "as", scope: !17, file: !2, line: 11, type: !23)
; CHECK: invalid set base type
!23 = !DIDerivedType(tag: DW_TAG_set_type, name: "SS", scope: !2, file: !2, line: 11, baseType: !24, size: 64, align: 64)
!24 = !DIBasicType(name: "SR", size: 8, encoding: DW_ATE_signed)
!25 = !DILocation(line: 11, scope: !17)
!26 = !DILocalVariable(name: "bs", scope: !17, file: !2, line: 11, type: !27)
!27 = !DIDerivedType(tag: DW_TAG_set_type, name: "ST", scope: !2, file: !2, line: 11, baseType: !23, size: 64, align: 64)
!28 = !DILocation(line: 17, scope: !17)
!29 = !DILocation(line: 18, scope: !17)
!30 = distinct !DISubprogram(name: "Main_M3", linkageName: "Main_M3", scope: !2, file: !2, line: 22, type: !31, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !20)
!31 = !DISubroutineType(types: !32)
!32 = !{!33, !35}
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ADDR", baseType: !34, size: 64, align: 64)
!34 = !DICompositeType(tag: DW_TAG_class_type, name: "ADDR__HeapObject", scope: !2, file: !2, line: 22, size: 64, align: 64, elements: !19, identifier: "AJWxb1")
!35 = !DIBasicType(name: "INTEGER", size: 64, encoding: DW_ATE_signed)
!36 = !DILocation(line: 23, scope: !30)
!37 = !DILocalVariable(name: "mode", arg: 1, scope: !30, file: !2, line: 22, type: !35)
!38 = !DILocation(line: 22, scope: !30)
