; RUN: opt -passes=declare-to-assign -S %s -o - \
; RUN: | FileCheck %s --check-prefix=WITHOUT-INTRINSIC

; RUN: sed 's/;Uncomment-with-sed//g' < %s \
; RUN: | opt -passes=declare-to-assign -S - -o - \
; RUN: | FileCheck %s --check-prefix=WITH-INTRINSIC

; WITHOUT-INTRINSIC-NOT: "debug-info-assignment-tracking"
; WITH-INTRINSIC: "debug-info-assignment-tracking"

;; Check that the module flag "debug-info-assignment-tracking" is only added if
;; declare-to-assign actually modifies the function. In the first run the
;; dbg.declare is commented out and the pass leaves the function unmodified.
;; In the second run sed is used to uncomment the dbg.declare and
;; declare-to-assign replaces it with a dbg.assign.

define dso_local void @f() !dbg !9 {
entry:
  %a = alloca i32, align 4
;Uncomment-with-sed  call void @llvm.dbg.declare(metadata ptr %a, metadata !13, metadata !DIExpression()), !dbg !16
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 17.0.0"}
!9 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 1, type: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 1, column: 12, scope: !9)
!16 = !DILocation(line: 1, column: 16, scope: !9)
!17 = !DILocation(line: 1, column: 19, scope: !9)
