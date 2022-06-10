; RUN: opt -early-cse -earlycse-debug-hash -S %s -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define i32 @foo() !dbg !6 {
entry:
  %0 = call i64 @llvm.ctpop.i64(i64 0), !dbg !14
  %1 = inttoptr i64 %0 to ptr, !dbg !14
  call void @llvm.dbg.value(metadata ptr %1, i64 0, metadata !11, metadata !13), !dbg !14
; CHECK: call void @llvm.dbg.value(metadata i64 0, metadata !11, metadata !DIExpression()), !dbg !13
  %call = call ptr (...) @baa(), !dbg !15
  %2 = ptrtoint ptr %call to i64, !dbg !16
  %3 = inttoptr i64 %2 to ptr, !dbg !16
  call void @llvm.dbg.value(metadata ptr %3, i64 0, metadata !11, metadata !13), !dbg !14
  %tobool = icmp ne ptr %3, null, !dbg !17
  br i1 %tobool, label %if.end, label %if.then, !dbg !19

if.then:                                          ; preds = %entry
  br label %cleanup, !dbg !20

if.end:                                           ; preds = %entry
  %4 = ptrtoint ptr %3 to i32, !dbg !21
  br label %cleanup, !dbg !22

cleanup:                                          ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %4, %if.end ], [ 0, %if.then ]
  ret i32 %retval.0, !dbg !22
}

declare ptr @baa(...)

; Function Attrs: nounwind readnone
declare i64 @llvm.ctpop.i64(i64)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/dir")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 6.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DILocalVariable(name: "ptr", scope: !6, file: !1, line: 4, type: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!13 = !DIExpression()
!14 = !DILocation(line: 4, column: 8, scope: !6)
!15 = !DILocation(line: 5, column: 9, scope: !6)
!16 = !DILocation(line: 5, column: 7, scope: !6)
!17 = !DILocation(line: 7, column: 7, scope: !18)
!18 = distinct !DILexicalBlock(scope: !6, file: !1, line: 7, column: 6)
!19 = !DILocation(line: 7, column: 6, scope: !6)
!20 = !DILocation(line: 8, column: 5, scope: !18)
!21 = !DILocation(line: 10, column: 10, scope: !6)
!22 = !DILocation(line: 11, column: 1, scope: !6)
