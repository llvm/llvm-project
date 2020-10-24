; RUN: verify-uselistorder %s

; Reproducer for PR36778.

; Verify that uses in metadata operands are considered when generating the
; use-list order. In this case the use-list order for @global_arr was not
; correctly preserved due to the uses in the dbg.value contant expressions not
; being considered, since they are wrapped in metadata.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = local_unnamed_addr global i64 0, align 8
@global_arr = common local_unnamed_addr global [10 x i64] zeroinitializer, align 16

define void @foo() local_unnamed_addr !dbg !6 {
entry:
  %0 = load i64, i64* getelementptr inbounds ([10 x i64], [10 x i64]* @global_arr, i64 0, i64 4), align 16
  call void @llvm.dbg.value(metadata i64* getelementptr inbounds ([10 x i64], [10 x i64]* @global_arr, i64 0, i64 5), metadata !10, metadata !DIExpression()), !dbg !13
  %1 = load i64, i64* getelementptr inbounds ([10 x i64], [10 x i64]* @global_arr, i64 0, i64 6), align 16
  call void @llvm.dbg.value(metadata i64* getelementptr inbounds ([10 x i64], [10 x i64]* @global_arr, i64 0, i64 6), metadata !10, metadata !DIExpression()), !dbg !14
  ret void
}

define void @bar() local_unnamed_addr !dbg !15 {
entry:
  call void @llvm.dbg.value(metadata i64* getelementptr inbounds ([10 x i64], [10 x i64]* @global_arr, i64 0, i64 7), metadata !17, metadata !DIExpression()), !dbg !18
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "uses.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 8.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 6, type: !7, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !0, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!10}
!10 = !DILocalVariable(name: "local1", scope: !6, file: !1, line: 7, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!13 = !DILocation(line: 8, column: 8, scope: !6)
!14 = !DILocation(line: 7, column: 9, scope: !6)
!15 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 12, type: !7, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: true, unit: !0, retainedNodes: !16)
!16 = !{!17}
!17 = !DILocalVariable(name: "local2", scope: !15, file: !1, line: 13, type: !11)
!18 = !DILocation(line: 14, column: 1, scope: !15)
