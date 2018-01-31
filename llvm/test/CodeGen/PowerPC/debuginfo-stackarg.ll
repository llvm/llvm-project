; RUN: llc < %s -O1 -stop-after=livedebugvalues -o - | FileCheck %s

; This ll-file was created by:
;   clang --target=powerpc-apple-darwin9 -O1 -S -g -emit-llvm debuginfo-stackarg.c
;
; with debuginfo-stackarg.c being the program:
;   long long foo(long long bar1, long long bar2, long long bar3, long long bar4, long long bar5)
;   {
;     return bar1 + bar2 + bar3 + bar4 + bar5;
;   }

; ModuleID = 'debuginfo-stackarg.c'
source_filename = "debuginfo-stackarg.c"
target datalayout = "E-m:o-p:32:32-f64:32:64-n32"
target triple = "powerpc-apple-macosx10.5.0"

; Function Attrs: nounwind readnone ssp uwtable
define i64 @foo(i64 %bar1, i64 %bar2, i64 %bar3, i64 %bar4, i64 %bar5) local_unnamed_addr #0 !dbg !8 {
; Variable bar5 should be associated with a position on the stack (offset relative r1).
; Let's verify that we point out the start address (lowest address).
;
; First find the metadata id for bar5.
; CHECK: !17 = !DILocalVariable(name: "bar5", arg: 5
;
; Now check that we got two entries on the fixed stack with "expected" offsets.
; CHECK-LABEL: fixedStack:
; CHECK: id: 0, type: default, offset: 60, size: 4
; CHECK: id: 1, type: default, offset: 56, size: 4
; CHECK-NOT: id: 2
; CHECK-LABEL: stack:
;
; Finally check the resulting function body.
; We expect to find a DBG_VALUE refering to the metadata id for bar5, using the lowest
; of the two fixed stack offsets found earlier.
; CHECK-LABEL: body:
; CHECK: DBG_VALUE $r1, 0, !17, !DIExpression(DW_OP_plus_uconst, 56)
entry:
  tail call void @llvm.dbg.value(metadata i64 %bar1, metadata !13, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i64 %bar2, metadata !14, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.value(metadata i64 %bar3, metadata !15, metadata !DIExpression()), !dbg !20
  tail call void @llvm.dbg.value(metadata i64 %bar4, metadata !16, metadata !DIExpression()), !dbg !21
  tail call void @llvm.dbg.value(metadata i64 %bar5, metadata !17, metadata !DIExpression()), !dbg !22
  %add = add nsw i64 %bar2, %bar1, !dbg !23
  %add1 = add nsw i64 %add, %bar3, !dbg !24
  %add2 = add nsw i64 %add1, %bar4, !dbg !25
  %add3 = add nsw i64 %add2, %bar5, !dbg !26
  ret i64 %add3, !dbg !27
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "debuginfo-stackarg.c", directory: "/repo")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !11, !11, !11, !11}
!11 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!12 = !{!13, !14, !15, !16, !17}
!13 = !DILocalVariable(name: "bar1", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!14 = !DILocalVariable(name: "bar2", arg: 2, scope: !8, file: !1, line: 1, type: !11)
!15 = !DILocalVariable(name: "bar3", arg: 3, scope: !8, file: !1, line: 1, type: !11)
!16 = !DILocalVariable(name: "bar4", arg: 4, scope: !8, file: !1, line: 1, type: !11)
!17 = !DILocalVariable(name: "bar5", arg: 5, scope: !8, file: !1, line: 1, type: !11)
!18 = !DILocation(line: 1, column: 25, scope: !8)
!19 = !DILocation(line: 1, column: 41, scope: !8)
!20 = !DILocation(line: 1, column: 57, scope: !8)
!21 = !DILocation(line: 1, column: 73, scope: !8)
!22 = !DILocation(line: 1, column: 89, scope: !8)
!23 = !DILocation(line: 3, column: 15, scope: !8)
!24 = !DILocation(line: 3, column: 22, scope: !8)
!25 = !DILocation(line: 3, column: 29, scope: !8)
!26 = !DILocation(line: 3, column: 36, scope: !8)
!27 = !DILocation(line: 3, column: 3, scope: !8)
