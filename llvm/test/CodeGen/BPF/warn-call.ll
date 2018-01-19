; RUN: not llc -march=bpfel < %s 2>&1 >/dev/null | FileCheck %s

; CHECK: error: warn_call.c
; CHECK: built-in function 'memcpy'
define i8* @warn(i8* returned, i8*, i64) local_unnamed_addr #0 !dbg !6 {
  tail call void @llvm.dbg.value(metadata i8* %0, i64 0, metadata !14, metadata !17), !dbg !18
  tail call void @llvm.dbg.value(metadata i8* %1, i64 0, metadata !15, metadata !17), !dbg !19
  tail call void @llvm.dbg.value(metadata i64 %2, i64 0, metadata !16, metadata !17), !dbg !20
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 %2, i1 false), !dbg !21
  %4 = tail call i8* @foo(i8* %0, i8* %1, i64 %2) #5, !dbg !22
  %5 = tail call fastcc i8* @bar(i8* %0), !dbg !23
  ret i8* %5, !dbg !24
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

declare i8* @foo(i8*, i8*, i64) local_unnamed_addr #2

; Function Attrs: noinline nounwind readnone
define internal fastcc i8* @bar(i8* readnone returned) unnamed_addr #3 !dbg !25 {
  tail call void @llvm.dbg.value(metadata i8* null, i64 0, metadata !28, metadata !17), !dbg !30
  tail call void @llvm.dbg.value(metadata i64 0, i64 0, metadata !29, metadata !17), !dbg !31
  ret i8* %0, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk 292174) (llvm/trunk 292179)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "warn_call.c", directory: "/w/llvm/bld")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 5.0.0 (trunk 292174) (llvm/trunk 292179)"}
!6 = distinct !DISubprogram(name: "warn", scope: !1, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !13)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !10, !12}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!12 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!13 = !{!14, !15, !16}
!14 = !DILocalVariable(name: "dst", arg: 1, scope: !6, file: !1, line: 4, type: !9)
!15 = !DILocalVariable(name: "src", arg: 2, scope: !6, file: !1, line: 4, type: !10)
!16 = !DILocalVariable(name: "len", arg: 3, scope: !6, file: !1, line: 4, type: !12)
!17 = !DIExpression()
!18 = !DILocation(line: 4, column: 18, scope: !6)
!19 = !DILocation(line: 4, column: 35, scope: !6)
!20 = !DILocation(line: 4, column: 54, scope: !6)
!21 = !DILocation(line: 6, column: 2, scope: !6)
!22 = !DILocation(line: 7, column: 2, scope: !6)
!23 = !DILocation(line: 8, column: 9, scope: !6)
!24 = !DILocation(line: 8, column: 2, scope: !6)
!25 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !7, isLocal: true, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !26)
!26 = !{!27, !28, !29}
!27 = !DILocalVariable(name: "dst", arg: 1, scope: !25, file: !1, line: 2, type: !9)
!28 = !DILocalVariable(name: "src", arg: 2, scope: !25, file: !1, line: 2, type: !10)
!29 = !DILocalVariable(name: "len", arg: 3, scope: !25, file: !1, line: 2, type: !12)
!30 = !DILocation(line: 2, column: 67, scope: !25)
!31 = !DILocation(line: 2, column: 86, scope: !25)
!32 = !DILocation(line: 2, column: 93, scope: !25)
