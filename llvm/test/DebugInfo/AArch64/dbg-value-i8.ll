; RUN: llc -stop-after=livedebugvars < %s | FileCheck %s
; ModuleID = 'debug_value-i8.bc'
source_filename = "debug_value.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; When ISel promotes the result of i8 add, the attached dbg.value shouldn't be dropped.
; CHECK: DBG_VALUE
; Function Attrs: nounwind
define dso_local i32 @bar(ptr, i32) local_unnamed_addr #0 !dbg !10 {
  %3 = trunc i32 %1 to i8, !dbg !20
  %4 = add i8 %3, 97, !dbg !20
  call void @llvm.dbg.value(metadata i8 %4, metadata !17, metadata !DIExpression()), !dbg !21
  %5 = zext i8 %4 to i32, !dbg !22
  tail call void @foo1(ptr %0, i32 %5) #2, !dbg !23
  tail call void @foo2(ptr %0, i8 %4) #2, !dbg !24
  ret i32 undef, !dbg !25
}

declare dso_local void @foo1(ptr, i32) local_unnamed_addr 

declare dso_local void @foo2(ptr, i8) local_unnamed_addr 

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "debug_value.c", directory: "./")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 7.0.0"}
!10 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 16, type: !11, isLocal: false, isDefinition: true, scopeLine: 16, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!5, !13, !5}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!14 = !{!15, !16, !17}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 16, type: !13)
!16 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !1, line: 16, type: !5)
!17 = !DILocalVariable(name: "t", scope: !10, file: !1, line: 17, type: !4)
!18 = !DILocation(line: 16, column: 14, scope: !10)
!19 = !DILocation(line: 16, column: 21, scope: !10)
!20 = !DILocation(line: 17, column: 11, scope: !10)
!21 = !DILocation(line: 17, column: 7, scope: !10)
!22 = !DILocation(line: 18, column: 10, scope: !10)
!23 = !DILocation(line: 18, column: 2, scope: !10)
!24 = !DILocation(line: 19, column: 2, scope: !10)
!25 = !DILocation(line: 20, column: 1, scope: !10)
