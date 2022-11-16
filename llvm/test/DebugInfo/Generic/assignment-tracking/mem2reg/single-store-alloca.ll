; RUN: opt -passes=mem2reg -S %s -o - -experimental-assignment-tracking \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Test assignment tracking debug info when mem2reg promotes a single-store
;; alloca. Additionally, check that all the dbg.assigns linked to the alloca
;; are cleaned up, including duplciates.

; CHECK: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %a, metadata ![[B:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %a, metadata ![[A:[0-9]+]], {{.*}}, metadata ptr undef
; CHECK-NEXT: ret

; CHECK-DAG: ![[A]] = !DILocalVariable(name: "a",
; CHECK-DAG: ![[B]] = !DILocalVariable(name: "b",

;; 1. using source:
;; $ cat test.cpp
;; int f(int a) { return a; }
;; 2. manually duplicating the dbg.assign lnked to a's alloca to ensure
;;    duplicates are still cleaned up,
;; 3. and with a dbg.assign for another variable ("b") attached to a's alloca
;;    to check that a dbg.value is generated for the variable to represent the
;;    store despite the store not having a dbg.assign linked for it.

; Function Attrs: mustprogress nounwind uwtable
define dso_local noundef i32 @_Z1fi(i32 noundef %a) #0 !dbg !7 {
entry:
  %a.addr = alloca i32, align 4, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !13, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !13, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.assign(metadata i1 undef, metadata !22, metadata !DIExpression(), metadata !13, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  store i32 %a, ptr %a.addr, align 4, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i32 %a, metadata !12, metadata !DIExpression(), metadata !19, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  %0 = load i32, ptr %a.addr, align 4, !dbg !20
  ret i32 %0, !dbg !21
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0)"}
!7 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 1, column: 23, scope: !7)
!21 = !DILocation(line: 1, column: 16, scope: !7)
!22 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 1, type: !10)
