; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Check that this testcase does not crash.
; CHECK: call foo0

target triple = "hexagon"

; Function Attrs: nounwind
declare void @foo0() local_unnamed_addr #0

; Function Attrs: nounwind
define void @foo1() local_unnamed_addr #0 !dbg !33 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !51, metadata !52), !dbg !53
  tail call void @foo0(), !dbg !54
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "disable-tail-calls"="true" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv5" "target-features"=",-hvx,-long-calls" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26, !27}
!llvm.linker.options = !{!29, !30, !31, !32, !29, !30, !31, !32, !29, !30, !31, !32, !29, !30, !31, !32, !29, !30, !31, !32, !29, !30, !31, !32}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !22)
!1 = !DIFile(filename: "foo.i", directory: "/path")
!2 = !{!3, !16}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !4, line: 122, size: 8, elements: !5)
!4 = !DIFile(filename: "foo.h", directory: "/path")
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!6 = !DIEnumerator(name: "E0", value: 7)
!7 = !DIEnumerator(name: "E1", value: 6)
!8 = !DIEnumerator(name: "E2", value: 5)
!9 = !DIEnumerator(name: "E3", value: 0)
!10 = !DIEnumerator(name: "E4", value: 1)
!11 = !DIEnumerator(name: "E5", value: 7)
!12 = !DIEnumerator(name: "E6", value: 5)
!13 = !DIEnumerator(name: "E7", value: 4)
!14 = !DIEnumerator(name: "E8", value: 4)
!15 = !DIEnumerator(name: "E9", value: 10)
!16 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !4, line: 136, size: 8, elements: !17)
!17 = !{!18, !19, !20, !21}
!18 = !DIEnumerator(name: "F0", value: 1)
!19 = !DIEnumerator(name: "F1", value: 2)
!20 = !DIEnumerator(name: "F2", value: 4)
!21 = !DIEnumerator(name: "F3", value: 7)
!22 = !{!23, !24, !25}
!23 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!24 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "t0_t", file: !4, line: 38, baseType: !24)
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 6, !"Linker Options", !28}
!28 = !{!29, !30, !31, !32}
!29 = !{!"foo0", !".text"}
!30 = !{!"foo1", !".text"}
!31 = !{!"foo2", !".text"}
!32 = !{!"foo3", !".text"}
!33 = distinct !DISubprogram(name: "foo1", scope: !34, file: !34, line: 84, type: !35, isLocal: false, isDefinition: true, scopeLine: 85, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !44)
!34 = !DIFile(filename: "foo.c", directory: "/path")
!35 = !DISubroutineType(types: !36)
!36 = !{!37, !38, !39, !40, !41, !42, !43, !37}
!37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 32)
!39 = !DIDerivedType(tag: DW_TAG_typedef, name: "t1_t", file: !4, line: 35, baseType: !23)
!40 = !DIDerivedType(tag: DW_TAG_typedef, name: "t2_t", file: !4, line: 36, baseType: !23)
!41 = !DIDerivedType(tag: DW_TAG_typedef, name: "t3_t", file: !4, line: 43, baseType: !23)
!42 = !DIDerivedType(tag: DW_TAG_typedef, name: "t4_t", file: !4, line: 133, baseType: !3)
!43 = !DIDerivedType(tag: DW_TAG_typedef, name: "t5_t", file: !4, line: 141, baseType: !16)
!44 = !{!45, !46, !47, !48, !49, !50, !51}
!45 = !DILocalVariable(name: "a0", arg: 1, scope: !33, file: !34, line: 84, type: !38)
!46 = !DILocalVariable(name: "a1", arg: 2, scope: !33, file: !34, line: 84, type: !39)
!47 = !DILocalVariable(name: "a2", arg: 3, scope: !33, file: !34, line: 84, type: !40)
!48 = !DILocalVariable(name: "a3", arg: 4, scope: !33, file: !34, line: 84, type: !41)
!49 = !DILocalVariable(name: "a4", arg: 5, scope: !33, file: !34, line: 84, type: !42)
!50 = !DILocalVariable(name: "a5", arg: 6, scope: !33, file: !34, line: 84, type: !43)
!51 = !DILocalVariable(name: "a6", arg: 7, scope: !33, file: !34, line: 84, type: !37)
!52 = !DIExpression()
!53 = !DILocation(line: 84, column: 169, scope: !33)
!54 = !DILocation(line: 86, column: 12, scope: !33)
