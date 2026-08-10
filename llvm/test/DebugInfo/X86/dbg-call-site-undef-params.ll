; RUN: llc -emit-call-site-info %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

;; Compiled from source:
;; extern int fn1 (long int x, long int y, long int z);
;; __attribute__((noinline)) long int
;; fn2 (long int a, long int b, long int c)
;; {
;;   long int q = 2 * a;
;;   return fn1 (5, 6, 7);
;; }
;; int main(void) {
;;         return fn2(14, 23, 34);
;; }
;; Using command:
;; clang -g -O2 m.c -emit-llvm -S -c -o m.ll

;; Verify that call site info is not created for parameters marked as "undef".
; CHECK: DW_TAG_GNU_call_site
; CHECK: DW_AT_abstract_origin ({{.*}} "fn2")
; CHECK-NOT: DW_TAG_GNU_call_site_parameter

; ModuleID = 'm.c'
source_filename = "m.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i64 @fn2(i64 %a, i64 %b, i64 %c) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i64 undef, metadata !12, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i64 undef, metadata !13, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i64 undef, metadata !14, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i64 undef, metadata !15, metadata !DIExpression()), !dbg !16
  %call = tail call i32 @fn1(i64 5, i64 6, i64 7) #4, !dbg !16
  %conv = sext i32 %call to i64, !dbg !16
  ret i64 %conv, !dbg !16
}

declare !dbg !19 dso_local i32 @fn1(i64, i64, i64) local_unnamed_addr

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr !dbg !23 {
entry:
  %call = tail call i64 @fn2(i64 undef, i64 undef, i64 undef), !dbg !26
  %conv = trunc i64 %call to i32, !dbg !26
  ret i32 %conv, !dbg !26
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "m.c", directory: "/dir")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 4, type: !8, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10, !10}
!10 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !15}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 4, type: !10)
!13 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !1, line: 4, type: !10)
!14 = !DILocalVariable(name: "c", arg: 3, scope: !7, file: !1, line: 4, type: !10)
!15 = !DILocalVariable(name: "q", scope: !7, file: !1, line: 6, type: !10)
!16 = !DILocation(line: 0, scope: !7)
!19 = !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !20, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{!22, !10, !10, !10}
!22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!23 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 9, type: !24, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!24 = !DISubroutineType(types: !25)
!25 = !{!22}
!26 = !DILocation(line: 10, column: 16, scope: !23)
