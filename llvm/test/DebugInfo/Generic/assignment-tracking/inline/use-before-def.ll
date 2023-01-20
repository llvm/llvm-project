; RUN: opt -passes=inline %s -S -o - \
; RUN: | FileCheck %s

;; Hand modified from:
;; $ cat test.c
;; int g = 5;
;; static int callee() {
;;   int local = g;
;;   return local;
;; }
;;
;; int fun() {
;;   return callee();
;; }
;;
;; IR grabbed before inlining in:
;;     $ clang++ -O2 -g
;; Then modified (see comment in body).

;; NOTE: Although this reproducer is contrived, this has been observed in real
;; builds with assignment tracking (dbg.assign intrinsics) - in fact, it caused
;; verifier failures in some cases using assignment tracking (good). This is
;; the simplest test I could write.

;; use-before-defs in debug intrinsics should be preserved through inlining,
;; or at the very least should be converted to undefs. The previous behaviour
;; was to replace the use-before-def operand with empty metadata, which signals
;; cleanup passes that it's okay to remove the debug intrinsic (bad). Check
;; that this no longer happens.

; CHECK: define dso_local i32 @fun()
; CHECK-NEXT: entry
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %0

@g = dso_local local_unnamed_addr global i32 5, align 4, !dbg !0

define dso_local i32 @fun() local_unnamed_addr #0 !dbg !11 {
entry:
  %call = call fastcc i32 @callee(), !dbg !14
  ret i32 %call, !dbg !15
}

define internal fastcc i32 @callee() unnamed_addr #1 !dbg !16 {
entry:
  ;; dbg.value moved here from after %0 def.
  call void @llvm.dbg.value(metadata i32 %0, metadata !18, metadata !DIExpression()), !dbg !24
  %0 = load i32, ptr @g, align 4, !dbg !19
  ret i32 %0, !dbg !25
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #2


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !1000}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
!11 = distinct !DISubprogram(name: "fun", scope: !3, file: !3, line: 7, type: !12, scopeLine: 7, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !DILocation(line: 8, column: 10, scope: !11)
!15 = !DILocation(line: 8, column: 3, scope: !11)
!16 = distinct !DISubprogram(name: "callee", scope: !3, file: !3, line: 2, type: !12, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!17 = !{!18}
!18 = !DILocalVariable(name: "local", scope: !16, file: !3, line: 3, type: !6)
!19 = !DILocation(line: 3, column: 15, scope: !16)
!24 = !DILocation(line: 0, scope: !16)
!25 = !DILocation(line: 4, column: 3, scope: !16)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
