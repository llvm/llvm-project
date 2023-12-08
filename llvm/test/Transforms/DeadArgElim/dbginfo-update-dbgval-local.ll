; RUN: opt -passes=deadargelim -S < %s | FileCheck %s

; Verify that the dbg.value intrinsics that use the dead argument and return
; value are marked as poison to indicate that the values are optimized out.

; Reproducer for PR23260.

; CHECK-LABEL: define internal void @bar()
; CHECK: call void @llvm.dbg.value(metadata i32 poison, metadata ![[LOCAL1:[0-9]+]]
; CHECK: call void @sink()

; Function Attrs: alwaysinline nounwind uwtable
define internal i32 @bar(i32 %deadarg) #1 !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata i32 %deadarg, metadata !15, metadata !DIExpression()), !dbg !17
  call void @sink(), !dbg !17
  ret i32 123, !dbg !17
}

; CHECK-LABEL: define void @foo()
; CHECK: call void @bar()
; CHECK: call void @llvm.dbg.value(metadata i32 poison, metadata ![[LOCAL2:[0-9]+]]
; CHECK: call void @bar()

; Function Attrs: nounwind uwtable
define void @foo() #0 !dbg !6 {
entry:
  %deadret = call i32 @bar(i32 0), !dbg !9
  call void @llvm.dbg.value(metadata i32 %deadret, metadata !16, metadata !DIExpression()), !dbg !9
  call i32 @bar(i32 1), !dbg !9
  ret void, !dbg !9
}

declare void @sink() local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable }
attributes #1 = { alwaysinline nounwind uwtable }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

; CHECK: ![[LOCAL1]] = !DILocalVariable(name: "local1"
; CHECK: ![[LOCAL2]] = !DILocalVariable(name: "local2"

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "pr23260.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 8.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 4, column: 3, scope: !6)
!10 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "local1", arg: 1, scope: !10, file: !1, line: 2, type: !13)
!16 = !DILocalVariable(name: "local2", arg: 1, scope: !6, file: !1, line: 2, type: !13)
!17 = !DILocation(line: 2, column: 52, scope: !10)
