; RUN: llc < %s -mtriple=thumbv7em-arm-none-eabi -O3

; When using -Oz and -g, this code generated an abort in the ARM load/store optimizer.

%struct.s = type { %struct.s* }

; Function Attrs: minsize nounwind optsize readonly
define %struct.s* @s_idx(%struct.s* readonly %xl) local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.s* %xl, metadata !17, metadata !18), !dbg !19
  br label %while.cond, !dbg !20

while.cond:                                       ; preds = %while.body, %entry
  %xl.addr.0 = phi %struct.s* [ %xl, %entry ], [ %0, %while.body ]
  %tobool = icmp eq %struct.s* %xl.addr.0, null
  br i1 %tobool, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %next = getelementptr inbounds %struct.s, %struct.s* %xl.addr.0, i32 0, i32 0
  %0 = load %struct.s*, %struct.s** %next, align 4
  tail call void @llvm.dbg.value(metadata %struct.s* %0, metadata !17, metadata !18), !dbg !19
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret %struct.s* null
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/a/b/c")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang version 4.0.0 "}
!8 = distinct !DISubprogram(name: "s_idx", scope: !1, file: !1, line: 6, type: !9, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !16)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "ezxml_t", file: !1, line: 1, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 32, align: 32)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 2, size: 32, align: 32, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !13, file: !1, line: 3, baseType: !11, size: 32, align: 32)
!16 = !{!17}
!17 = !DILocalVariable(name: "xl", arg: 1, scope: !8, file: !1, line: 6, type: !11)
!18 = !DIExpression()
!19 = !DILocation(line: 6, column: 27, scope: !8)
!20 = !DILocation(line: 8, column: 5, scope: !8)
