; RUN: llc -global-isel=1 -filetype=obj -mtriple=aarch64-linux-gnu -o - %s | llvm-dwarfdump -v - | FileCheck %s
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location


%struct.mystruct = type { double, double, double, double, double }

@.str = private unnamed_addr constant [5 x i8] c"%llu\00", align 1
define dso_local void @foo(%struct.mystruct* %ms) !dbg !9 {
entry:
  call void @llvm.dbg.declare(metadata %struct.mystruct* %ms, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = ptrtoint %struct.mystruct* %ms to i64, !dbg !23
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i64 %0), !dbg !24
  ret void, !dbg !25
}
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare dso_local i32 @printf(i8*, ...)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: ".", directory: ".")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!5 = !{i32 7, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 10.0.0"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !10, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "mystruct", file: !1, line: 3, baseType: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 1, size: 320, elements: !14)
!14 = !{!15, !17, !18, !19, !20}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 2, baseType: !16, size: 64)
!16 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !1, line: 2, baseType: !16, size: 64, offset: 64)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !13, file: !1, line: 2, baseType: !16, size: 64, offset: 128)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !13, file: !1, line: 2, baseType: !16, size: 64, offset: 192)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !13, file: !1, line: 2, baseType: !16, size: 64, offset: 256)
!21 = !DILocalVariable(name: "ms", arg: 1, scope: !9, file: !1, line: 5, type: !12)
!22 = !DILocation(line: 5, column: 19, scope: !9)
!23 = !DILocation(line: 7, column: 1, scope: !9)
!24 = !DILocation(line: 6, column: 5, scope: !9)
!25 = !DILocation(line: 8, column: 1, scope: !9)
