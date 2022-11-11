; RUN: opt -S -licm < %s | FileCheck %s

; LICM is trying to merge the two `store` in block %14 and %17, but given their
; locations disagree, it sets a line zero location instead instead of picking a
; random one (the DILocation picked the nearest enclosing scope of the two stores).

; Original C testcase.
; volatile int a;
; extern int g;
; int g;
; void f1() { 
;  while (a) { 
;    g = 0;
;    if (a)
;      g = 0;
;  } 
; }

; CHECK: bb22:
; CHECK-NEXT: store i32 0, ptr getelementptr inbounds ([2 x i32], ptr @g_390, i64 0, i64 1), align 4, !dbg [[storeLocation:![0-9]+]] 
; CHECK: [[storeLocation]] = !DILocation(line: 0

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@b = global i32 0, align 4, !dbg !0
@c = global i32 0, align 4, !dbg !10
@g_390 = local_unnamed_addr global [2 x i32] zeroinitializer, align 4, !dbg !12
@a = local_unnamed_addr global i32 0, align 4, !dbg !6

define i32 @main() local_unnamed_addr !dbg !22 {
bb:
  %i = load volatile i32, ptr @b, align 4, !dbg !37, !tbaa !40
  %i1 = icmp sgt i32 %i, -9, !dbg !44
  br i1 %i1, label %bb2, label %bb4, !dbg !45

bb2:                                              ; preds = %bb
  br label %bb8, !dbg !45

bb3:                                              ; preds = %bb8
  br label %bb4, !dbg !45

bb4:                                              ; preds = %bb3, %bb
  %i5 = load volatile i32, ptr @c, align 4, !dbg !46, !tbaa !40
  %i6 = icmp slt i32 %i5, 6, !dbg !47
  br i1 %i6, label %bb7, label %bb23, !dbg !48

bb7:                                              ; preds = %bb4
  br label %bb13, !dbg !48

bb8:                                              ; preds = %bb8, %bb2
  %i9 = load volatile i32, ptr @b, align 4, !dbg !49, !tbaa !40
  %i10 = add nsw i32 %i9, -1, !dbg !49
  store volatile i32 %i10, ptr @b, align 4, !dbg !49, !tbaa !40
  %i11 = load volatile i32, ptr @b, align 4, !dbg !37, !tbaa !40
  %i12 = icmp sgt i32 %i11, -9, !dbg !44
  br i1 %i12, label %bb8, label %bb3, !dbg !45, !llvm.loop !50

bb13:                                             ; preds = %bb17, %bb7
  store i32 0, ptr getelementptr inbounds ([2 x i32], ptr @g_390, i64 0, i64 1), align 4, !dbg !53, !tbaa !40
  %i14 = load volatile i32, ptr @b, align 4, !dbg !54, !tbaa !40
  %i15 = icmp eq i32 %i14, 0, !dbg !54
  br i1 %i15, label %bb16, label %bb17, !dbg !55

bb16:                                             ; preds = %bb13
  store i32 0, ptr getelementptr inbounds ([2 x i32], ptr @g_390, i64 0, i64 1), align 4, !dbg !56, !tbaa !40
  br label %bb17

bb17:                                             ; preds = %bb16, %bb13
  %i18 = load volatile i32, ptr @c, align 4, !dbg !57, !tbaa !40
  %i20 = load volatile i32, ptr @c, align 4, !dbg !46, !tbaa !40
  %i21 = icmp slt i32 %i20, 6, !dbg !47
  br i1 %i21, label %bb13, label %bb22, !dbg !48, !llvm.loop !58

bb22:                                             ; preds = %bb17
  br label %bb23, !dbg !48

bb23:                                             ; preds = %bb22, %bb4
  ret i32 0, !dbg !60
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project d3588d0814c4cbc7fca677b4d9634f6e1428a331)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None, sysroot: "/")
!3 = !DIFile(filename: "a.c", directory: "/Users/davide/work/build/bin")
!4 = !{}
!5 = !{!6, !0, !10, !12}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "g_390", scope: !2, file: !3, line: 2, type: !14, isLocal: false, isDefinition: true)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 64, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 2)
!17 = !{i32 7, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 7, !"PIC Level", i32 2}
!21 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project d3588d0814c4cbc7fca677b4d9634f6e1428a331)"}
!22 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 3, type: !23, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !25)
!23 = !DISubroutineType(types: !24)
!24 = !{!9}
!25 = !{!26}
!26 = !DILocalVariable(name: "l_1546", scope: !27, file: !3, line: 11, type: !32)
!27 = distinct !DILexicalBlock(scope: !28, file: !3, line: 10, column: 10)
!28 = distinct !DILexicalBlock(scope: !29, file: !3, line: 8, column: 9)
!29 = distinct !DILexicalBlock(scope: !30, file: !3, line: 6, column: 23)
!30 = distinct !DILexicalBlock(scope: !31, file: !3, line: 6, column: 3)
!31 = distinct !DILexicalBlock(scope: !22, file: !3, line: 6, column: 3)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !33, size: 288, elements: !34)
!33 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!34 = !{!35, !36, !35}
!35 = !DISubrange(count: 3)
!36 = !DISubrange(count: 4)
!37 = !DILocation(line: 4, column: 10, scope: !38)
!38 = distinct !DILexicalBlock(scope: !39, file: !3, line: 4, column: 3)
!39 = distinct !DILexicalBlock(scope: !22, file: !3, line: 4, column: 3)
!40 = !{!41, !41, i64 0}
!41 = !{!"int", !42, i64 0}
!42 = !{!"omnipotent char", !43, i64 0}
!43 = !{!"Simple C/C++ TBAA"}
!44 = !DILocation(line: 4, column: 12, scope: !38)
!45 = !DILocation(line: 4, column: 3, scope: !39)
!46 = !DILocation(line: 6, column: 10, scope: !30)
!47 = !DILocation(line: 6, column: 12, scope: !30)
!48 = !DILocation(line: 6, column: 3, scope: !31)
!49 = !DILocation(line: 4, column: 19, scope: !38)
!50 = distinct !{!50, !45, !51, !52}
!51 = !DILocation(line: 5, column: 5, scope: !39)
!52 = !{!"llvm.loop.unroll.disable"}
!53 = !DILocation(line: 7, column: 14, scope: !29)
!54 = !DILocation(line: 8, column: 9, scope: !28)
!55 = !DILocation(line: 8, column: 9, scope: !29)
!56 = !DILocation(line: 12, column: 16, scope: !27)
!57 = !DILocation(line: 6, column: 19, scope: !30)
!58 = distinct !{!58, !48, !59, !52}
!59 = !DILocation(line: 14, column: 3, scope: !31)
!60 = !DILocation(line: 15, column: 1, scope: !22)
