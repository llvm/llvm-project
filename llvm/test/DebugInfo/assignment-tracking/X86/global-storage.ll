; RUN: llc %s -stop-after=finalize-isel -o - | FileCheck %s


; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - | FileCheck %s

;; Local variable has global storage. Check AssignmentTrackingAnalysis doesn't
;; crash/assert.

;; FIXME: We ideally want a DBG_VALUE deref here. It's not possible with the
;; current setup, but will be possible when assignment tracking is extended to
;; understand non-alloca storage.

; CHECK: stack: []
; CHECK-NOT: DBG_

target triple = "x86_64-unknown-linux-gnu"

@a = dso_local global i32 0, align 4

define dso_local void @_Z3funi(i32 noundef %x) !dbg !15 {
entry:
  store i32 %x, ptr @a, align 4, !DIAssignID !27
  call void @llvm.dbg.assign(metadata i32 %x, metadata !23, metadata !DIExpression(), metadata !27, metadata ptr @a, metadata !DIExpression()), !dbg !21
  ret void
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !13}
!llvm.ident = !{!14}

!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!14 = !{!"clang version 17.0.0"}
!15 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funi", scope: !3, file: !3, line: 2, type: !16, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !5}
!18 = !{}
!20 = !DILocalVariable(name: "x", arg: 1, scope: !15, file: !3, line: 2, type: !5)
!21 = !DILocation(line: 0, scope: !15)
!23 = !DILocalVariable(name: "a", scope: !15, file: !3, line: 3, type: !5)
!27 = distinct !DIAssignID()
