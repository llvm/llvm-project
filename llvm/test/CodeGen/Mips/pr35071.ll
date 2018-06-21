; RUN: llc -mtriple mips64-unknown-freebsd12.0 -relocation-model pic -mcpu=mips4 -target-abi n64 -O2 -verify-machineinstrs -o - %s

; Test that the long branch pass does not crash due to the control flow
; optimizer producing malformed basic block operands due to the backend
; failing to handle debug information around branch instructions.

define void @f() !dbg !5 {
entry:
  %cmp = icmp eq i32 undef, 0, !dbg !16
  %conv = zext i1 %cmp to i32, !dbg !16
  tail call void @llvm.dbg.value(metadata i32 %conv, metadata !11, metadata !DIExpression()), !dbg !17
  %tobool = icmp eq i32 undef, 0, !dbg !18
  br i1 %tobool, label %if.end, label %cleanup7.critedge, !dbg !21

if.end:                                           ; preds = %entry
  %call6 = call i32 bitcast (i32 (...)* @j to i32 (i32)*)(i32 signext %conv)
#4, !dbg !22
  br label %cleanup7, !dbg !23

cleanup7.critedge:                                ; preds = %entry
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull undef) #4, !dbg !24
  br label %cleanup7

cleanup7:                                         ; preds = %cleanup7.critedge,
  ret void
}

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare i32 @j(...)

declare void @llvm.dbg.value(metadata, metadata, metadata) #3
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang
version 6.0.0", isOptimized: true, runtimeVersion:
0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename:
"/tmp//<stdin>", directory:
"/tmp/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DISubprogram(name: "f", scope: !6, file: !6, line: 8, type: !7,
isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !0,
variables: !10)
!6 = !DIFile(filename:
"/tmp/test.c",
directory: "/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12, !14}
!11 = !DILocalVariable(name: "e", scope: !5, file: !6, line: 9, type: !9)
!12 = !DILocalVariable(name: "g", scope: !13, file: !6, line: 11, type: !9)
!13 = distinct !DILexicalBlock(scope: !5, file: !6, line: 10, column: 3)
!14 = !DILocalVariable(name: "d", scope: !13, file: !6, line: 12, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "a", file: !6, line: 2,
baseType: !9)
!16 = !DILocation(line: 9, column: 15, scope: !5)
!17 = !DILocation(line: 9, column: 7, scope: !5)
!18 = !DILocation(line: 12, column: 5, scope: !19)
!19 = distinct !DILexicalBlock(scope: !20, file: !6, line: 12, column: 5)
!20 = distinct !DILexicalBlock(scope: !5, file: !6, line: 10, column: 3)
!21 = !DILocation(line: 12, column: 5, scope: !20)
!22 = !DILocation(line: 16, column: 3, scope: !5)
!23 = !DILocation(line: 17, column: 1, scope: !5)
!24 = !DILocation(line: 15, column: 3, scope: !5)
