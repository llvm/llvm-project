;; llc will generate additional 'empty' DW_TAG_subroutine in sum.c's CU.
;; It will not be considered by the statistics.
; RUN: llc %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s

;; Instructions to regenerate IR:
;; clang -g -flto -emit-llvm -S -o main.ll -c main.c
;; clang -g -flto -emit-llvm -S -o sum.ll -c sum.c
;; llvm-link -S -o linked.ll main.ll sum.ll
;; opt -O1 linked.ll -S -o merged.ll
;; Hard coded a call to llvm.dbg.value intrinsic, replacing %10 argument with undef, in order to have 0% location coverage for a CCU referencing DIE.

;; Source files:
;;main.c:
;;extern int sum(int a, int b);
;;
;;int main()
;;{
;;	int a = 10, b = 5;
;;	int c = sum(a,b);
;; int d = c + sum(c,2);
;;	return 0;
;;}
;;sum.c:
;;__attribute__((always_inline)) int sum(int a, int b)
;;{
;;	int result = a + b;
;;	return result;
;;}

; CHECK:      "#source variables with location": 10,
; CHECK:      "#variables with 0% of parent scope covered by DW_AT_location": 1,
; CHECK:      "#params with 0% of parent scope covered by DW_AT_location": 1,

; ModuleID = 'linked.ll'
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !11 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !15, metadata !DIExpression()), !dbg !16
  store i32 10, i32* %2, align 4, !dbg !16
  call void @llvm.dbg.declare(metadata i32* %3, metadata !17, metadata !DIExpression()), !dbg !16
  store i32 5, i32* %3, align 4, !dbg !16
  call void @llvm.dbg.declare(metadata i32* %4, metadata !19, metadata !DIExpression()), !dbg !16
  %6 = load i32, i32* %2, align 4, !dbg !16
  %7 = load i32, i32* %3, align 4, !dbg !16
  call void @llvm.dbg.value(metadata i32 %6, metadata !23, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %7, metadata !29, metadata !DIExpression()), !dbg !27
  %8 = add nsw i32 %7, %6, !dbg !27
  call void @llvm.dbg.value(metadata i32 %8, metadata !31, metadata !DIExpression()), !dbg !27
  store i32 %8, i32* %4, align 4, !dbg !16
  call void @llvm.dbg.declare(metadata i32* %5, metadata !32, metadata !DIExpression()), !dbg !16
  %9 = load i32, i32* %4, align 4, !dbg !16
  %10 = load i32, i32* %4, align 4, !dbg !16
  call void @llvm.dbg.value(metadata i32 undef, metadata !23, metadata !DIExpression()), !dbg !36 ;; Hard coded line: There was %10 instead of undef.
  call void @llvm.dbg.value(metadata i32 2, metadata !29, metadata !DIExpression()), !dbg !36
  %11 = add nsw i32 2, %10, !dbg !36
  call void @llvm.dbg.value(metadata i32 %11, metadata !31, metadata !DIExpression()), !dbg !36
  %12 = add nsw i32 %9, %11, !dbg !16
  store i32 %12, i32* %5, align 4, !dbg !16
  ret i32 0, !dbg !16
}

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local i32 @sum(i32 %0, i32 %1) local_unnamed_addr #2 !dbg !24 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !23, metadata !DIExpression()), !dbg !41
  call void @llvm.dbg.value(metadata i32 %1, metadata !29, metadata !DIExpression()), !dbg !41
  %3 = add nsw i32 %1, %0, !dbg !41
  call void @llvm.dbg.value(metadata i32 %3, metadata !31, metadata !DIExpression()), !dbg !41
  ret i32 %3, !dbg !41
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { noinline nounwind optnone uwtable }
attributes #2 = { alwaysinline mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn }

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.c", directory: "/dir")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !4, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "sum.c", directory: "/dir")
!5 = !{!"clang version 14.0.0"}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !12, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocalVariable(name: "a", scope: !11, file: !1, line: 5, type: !14)
!16 = !DILocation(line: 5, column: 6, scope: !11)
!17 = !DILocalVariable(name: "b", scope: !11, file: !1, line: 5, type: !14)
!19 = !DILocalVariable(name: "c", scope: !11, file: !1, line: 6, type: !14)
!23 = !DILocalVariable(name: "a", arg: 1, scope: !24, file: !4, line: 1, type: !14)
!24 = distinct !DISubprogram(name: "sum", scope: !4, file: !4, line: 1, type: !25, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !2)
!25 = !DISubroutineType(types: !26)
!26 = !{!14, !14, !14}
!27 = !DILocation(line: 0, scope: !24, inlinedAt: !28)
!28 = distinct !DILocation(line: 6, column: 10, scope: !11)
!29 = !DILocalVariable(name: "b", arg: 2, scope: !24, file: !4, line: 1, type: !14)
!31 = !DILocalVariable(name: "result", scope: !24, file: !4, line: 3, type: !14)
!32 = !DILocalVariable(name: "d", scope: !11, file: !1, line: 7, type: !14)
!36 = !DILocation(line: 0, scope: !24, inlinedAt: !37)
!37 = distinct !DILocation(line: 7, column: 14, scope: !11)
!41 = !DILocation(line: 0, scope: !24)
