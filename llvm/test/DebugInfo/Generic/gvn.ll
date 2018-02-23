; RUN: opt < %s -gvn -S | FileCheck %s
;
; Produced at -O2 from:
; int a, b;
; void f1(int *p1) {
;   if (b)
;     a = 1;
;   if (a && *p1)
;     f4();
; }
; int f2(int);
; void f3(void) {
;   a = f2(1);
;   f1(&a);
; }
source_filename = "test/DebugInfo/Generic/gvn.ll"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

@a = common global i32 0, align 4, !dbg !0
@b = common global i32 0, align 4, !dbg !6

; Function Attrs: nounwind
define void @f3() #0 !dbg !12 {
entry:
  ; Verify that the call still has a debug location after GVN.
  ; CHECK: %call = tail call i32 @f2(i32 1) #{{[0-9]}}, !dbg
  %call = tail call i32 @f2(i32 1) #0, !dbg !15
  store i32 %call, i32* @a, align 4, !dbg !15, !tbaa !24
  tail call void @llvm.dbg.value(metadata i32* @a, i64 0, metadata !22, metadata !28) #0, !dbg !29
  %0 = load i32, i32* @b, align 4, !dbg !29, !tbaa !24
  %tobool.i = icmp eq i32 %0, 0, !dbg !29
  br i1 %tobool.i, label %if.end.i, label %land.lhs.true.i.thread, !dbg !30

land.lhs.true.i.thread:                           ; preds = %entry
  store i32 1, i32* @a, align 4, !dbg !32, !tbaa !24
  br label %if.then.3.i, !dbg !33

  ; This instruction has no debug location -- in this
  ; particular case it was removed by a bug in SimplifyCFG.
if.end.i:                                         ; preds = %entry
  %.pr = load i32, i32* @a, align 4
  ; GVN is supposed to replace the load of %.pr with a direct reference to %call.
  ; CHECK: %tobool2.i = icmp eq i32 %call, 0, !dbg
  %tobool2.i = icmp eq i32 %.pr, 0, !dbg !34
  br i1 %tobool2.i, label %f1.exit, label %if.then.3.i, !dbg !34

if.then.3.i:                                      ; preds = %if.end.i, %land.lhs.true.i.thread
  %call.i = tail call i32 bitcast (i32 (...)* @f4 to i32 ()*)() #0, !dbg !35
  br label %f1.exit, !dbg !35

f1.exit:                                          ; preds = %if.then.3.i, %if.end.i
  ret void, !dbg !36
}

declare i32 @f2(i32)

declare i32 @f4(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.8.0 (trunk 245562) (llvm/trunk 245569)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7)
!7 = !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.8.0 (trunk 245562) (llvm/trunk 245569)"}
!12 = distinct !DISubprogram(name: "f3", scope: !3, file: !3, line: 9, type: !13, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !4)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DILocation(line: 5, scope: !16, inlinedAt: !23)
!16 = distinct !DILexicalBlock(scope: !17, file: !3, line: 5)
!17 = distinct !DISubprogram(name: "f1", scope: !3, file: !3, line: 2, type: !18, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !21)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!21 = !{!22}
!22 = !DILocalVariable(name: "p1", arg: 1, scope: !17, file: !3, line: 2, type: !20)
!23 = distinct !DILocation(line: 11, scope: !12)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !DIExpression()
!29 = !DILocation(line: 2, scope: !17, inlinedAt: !23)
!30 = !DILocation(line: 3, scope: !31, inlinedAt: !23)
!31 = distinct !DILexicalBlock(scope: !17, file: !3, line: 3)
!32 = !DILocation(line: 3, scope: !17, inlinedAt: !23)
!33 = !DILocation(line: 4, scope: !31, inlinedAt: !23)
!34 = !DILocation(line: 5, scope: !17, inlinedAt: !23)
!35 = !DILocation(line: 6, scope: !16, inlinedAt: !23)
!36 = !DILocation(line: 12, scope: !12)

