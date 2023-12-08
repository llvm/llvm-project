;
; RUN: %llc_dwarf -filetype=obj -O0 -dwarf-linkage-names=All < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

; Make sure we can handle enums with the same identifier but in enum types of
; different compile units.
; rdar://17628609

; CHECK: DW_TAG_compile_unit
; CHECK: 0x[[ENUM:.*]]: DW_TAG_enumeration_type
; CHECK-NEXT:   DW_AT_name {{.*}}"EA"
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_MIPS_linkage_name {{.*}}"_Z4topA2EA"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_type [DW_FORM_ref4] (cu + 0x{{.*}} => {0x[[ENUM]]}

; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_MIPS_linkage_name {{.*}}"_Z4topB2EA"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[ENUM]]

; Function Attrs: nounwind ssp uwtable
define void @_Z4topA2EA(i32 %sa) #0 !dbg !7 {
entry:
  %sa.addr = alloca i32, align 4
  store i32 %sa, ptr %sa.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %sa.addr, metadata !22, metadata !DIExpression()), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_Z4topB2EA(i32 %sa) #0 !dbg !17 {
entry:
  %sa.addr = alloca i32, align 4
  store i32 %sa, ptr %sa.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %sa.addr, metadata !25, metadata !DIExpression()), !dbg !26
  ret void, !dbg !27
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0, !12}
!llvm.module.flags = !{!19, !20}
!llvm.ident = !{!21, !21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (trunk 214102:214133) (llvm/trunk 214102:214132)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !11, imports: !11)
!1 = !DIFile(filename: "a.cpp", directory: "")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EA", line: 1, size: 32, align: 32, file: !1, elements: !4, identifier: "_ZTS2EA")
!4 = !{!5}
!5 = !DIEnumerator(name: "EA_0", value: 0) ; [ DW_TAG_enumerator ] [EA_0 :: 0]
!7 = distinct !DISubprogram(name: "topA", linkageName: "_Z4topA2EA", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !1, scope: !8, type: !9, retainedNodes: !11)
!8 = !DIFile(filename: "a.cpp", directory: "")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !3}
!11 = !{}
!12 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (trunk 214102:214133) (llvm/trunk 214102:214132)", isOptimized: false, emissionKind: FullDebug, file: !13, enums: !14, retainedTypes: !14, globals: !11, imports: !11)
!13 = !DIFile(filename: "b.cpp", directory: "")
!14 = !{!15}
!15 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "EA", line: 1, size: 32, align: 32, file: !13, elements: !4, identifier: "_ZTS2EA")
!17 = distinct !DISubprogram(name: "topB", linkageName: "_Z4topB2EA", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !12, scopeLine: 5, file: !13, scope: !18, type: !9, retainedNodes: !11)
!18 = !DIFile(filename: "b.cpp", directory: "")
!19 = !{i32 2, !"Dwarf Version", i32 3}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{!"clang version 3.5.0 (trunk 214102:214133) (llvm/trunk 214102:214132)"}
!22 = !DILocalVariable(name: "sa", line: 5, arg: 1, scope: !7, file: !8, type: !3)
!23 = !DILocation(line: 5, column: 14, scope: !7)
!24 = !DILocation(line: 6, column: 1, scope: !7)
!25 = !DILocalVariable(name: "sa", line: 5, arg: 1, scope: !17, file: !18, type: !3)
!26 = !DILocation(line: 5, column: 14, scope: !17)
!27 = !DILocation(line: 6, column: 1, scope: !17)
