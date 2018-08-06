; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -filetype=obj | llvm-dwarfdump - | FileCheck %s
; CHECK: 0x00000[[G:.*]]:     DW_TAG_variable
; CHECK-NEXT:                DW_AT_name	("g")
; CHECK: DW_TAG_array_type
; CHECK-NEXT:  DW_AT_type	({{.*}} "int")
; CHECK-NOT: DW_TAG
; CHECK:       DW_TAG_subrange_type
; CHECK-NEXT:     DW_AT_type	({{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:      DW_AT_count	(0x00000[[G]])
; Test that a VLA referring to a global variable is handled correctly.
; Clang doesn't generate this, but the verifier allows it.
source_filename = "/tmp/test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

@g = common local_unnamed_addr global i32 0, align 4, !dbg !0

define void @f() !dbg !12 {
entry:
  %0 = load i32, i32* @g, align 4, !dbg !22
  %1 = zext i32 %0 to i64, !dbg !22
  %vla = alloca i32, i64 %1, align 16, !dbg !22
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !16, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 2, metadata !21, metadata !DIExpression()), !dbg !22
  %call = call i32 (i32*, ...) bitcast (i32 (...)* @use to i32 (i32*, ...)*)(i32* nonnull %vla), !dbg !22
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

declare i32 @use(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 324259) (llvm/trunk 324261)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "/tmp/test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"PIC Level", i32 2}
!11 = !{!"clang version 7.0.0 (trunk 324259) (llvm/trunk 324261)"}
!12 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 3, type: !13, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{!16, !21}
!16 = !DILocalVariable(name: "array", scope: !12, file: !3, line: 4, type: !17)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, elements: !19)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !{!20}
!20 = !DISubrange(count: !1)
!21 = !DILocalVariable(name: "count", scope: !12, file: !3, line: 5, type: !18)
!22 = !DILocation(line: 7, column: 1, scope: !12)
