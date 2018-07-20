; RUN: opt -S -codegenprepare < %s | FileCheck %s
;
; This test case was generated from the following source code:
; 
; long long foo(int *ptr, int cond) {
;   long long result = 3;
;   unsigned val = *ptr; // line 3
;   switch (cond) {
;   case 3:
;     result = val;      // line 6
;     break;
;   case 4:
;     result += 2;
;   }
; 
;   return result + val;
; };
;
; When CGP moves a zext Z of a load L to the block where L lives,  Z should not
; retain its original debug location. Instead, Z should reuse the debug location
; associated with L. Logically the zero extend will become part of the load; the
; code generator will attempt to fuse the two instructions into a zextload.

; CHECK-LABEL: @test
; CHECK:   [[LOADVAL:%[0-9]+]] = load i32, i32* %ptr, align 4, !dbg [[DEBUGLOC:![0-9]+]]
; CHECK-NEXT:                    zext i32 [[LOADVAL]] to i64, !dbg [[DEBUGLOC]]
; CHECK:   [[DEBUGLOC]] = !DILocation(line: 3

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @test(i32* %ptr, i32 %cond) !dbg !5 {
entry:
  %0 = load i32, i32* %ptr, align 4, !dbg !7
  switch i32 %cond, label %sw.epilog [
    i32 3, label %sw.bb
    i32 4, label %sw.bb1
  ], !dbg !8

sw.bb:                                            ; preds = %entry
  %conv = zext i32 %0 to i64, !dbg !9
  br label %sw.epilog, !dbg !10

sw.bb1:                                           ; preds = %entry
  br label %sw.epilog, !dbg !11

sw.epilog:                                        ; preds = %sw.bb1, %entry, %sw.bb
  %result.0 = phi i64 [ 3, %entry ], [ 5, %sw.bb1 ], [ %conv, %sw.bb ]
  %conv2 = zext i32 %0 to i64, !dbg !12
  %add3 = add nuw nsw i64 %result.0, %conv2, !dbg !13
  ret i64 %add3, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 3, column: 18, scope: !5)
!8 = !DILocation(line: 4, column: 3, scope: !5)
!9 = !DILocation(line: 6, column: 14, scope: !5)
!10 = !DILocation(line: 7, column: 5, scope: !5)
!11 = !DILocation(line: 10, column: 3, scope: !5)
!12 = !DILocation(line: 12, column: 19, scope: !5)
!13 = !DILocation(line: 12, column: 17, scope: !5)
!14 = !DILocation(line: 12, column: 3, scope: !5)
