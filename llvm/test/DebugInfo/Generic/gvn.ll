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
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

@a = common global i32 0, align 4
@b = common global i32 0, align 4

; Function Attrs: nounwind
define void @f3() #0 !dbg !12 {
entry:
  ; Verify that the call still has a debug location after GVN.
  ; CHECK: %call = tail call i32 @f2(i32 1) #{{[0-9]}}, !dbg
  %call = tail call i32 @f2(i32 1) #3, !dbg !36
  store i32 %call, i32* @a, align 4, !dbg !36, !tbaa !25
  tail call void @llvm.dbg.value(metadata i32* @a, i64 0, metadata !11, metadata !21) #3, !dbg !39
  %0 = load i32, i32* @b, align 4, !dbg !39, !tbaa !25
  %tobool.i = icmp eq i32 %0, 0, !dbg !39
  br i1 %tobool.i, label %if.end.i, label %land.lhs.true.i.thread, !dbg !40

land.lhs.true.i.thread:                           ; preds = %entry
  store i32 1, i32* @a, align 4, !dbg !41, !tbaa !25
  br label %if.then.3.i, !dbg !42

if.end.i:                                         ; preds = %entry
  ; This instruction has no debug location -- in this
  ; particular case it was removed by a bug in SimplifyCFG.
  %.pr = load i32, i32* @a, align 4

  ; GVN is supposed to replace the load of %.pr with a direct reference to %call.
  ; CHECK: %tobool2.i = icmp eq i32 %call, 0, !dbg
  %tobool2.i = icmp eq i32 %.pr, 0, !dbg !43
  br i1 %tobool2.i, label %f1.exit, label %if.then.3.i, !dbg !43

if.then.3.i:                                      ; preds = %if.end.i, %land.lhs.true.i.thread
  %call.i = tail call i32 bitcast (i32 (...)* @f4 to i32 ()*)() #3, !dbg !44
  br label %f1.exit, !dbg !44

f1.exit:                                          ; preds = %if.end.i, %if.then.3.i
  ret void, !dbg !45
}

declare i32 @f2(i32)
declare i32 @f4(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 245562) (llvm/trunk 245569)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !15)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!4 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DILocalVariable(name: "p1", arg: 1, scope: !4, file: !1, line: 2, type: !8)
!12 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 9, type: !13, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{!16, !17}
!16 = !DIGlobalVariable(name: "a", scope: !0, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, variable: i32* @a)
!17 = !DIGlobalVariable(name: "b", scope: !0, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, variable: i32* @b)
!18 = !{i32 2, !"Dwarf Version", i32 2}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{!"clang version 3.8.0 (trunk 245562) (llvm/trunk 245569)"}
!21 = !DIExpression()
!22 = !DILocation(line: 2, scope: !4)
!23 = !DILocation(line: 3, scope: !24)
!24 = distinct !DILexicalBlock(scope: !4, file: !1, line: 3)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 3, scope: !4)
!30 = !DILocation(line: 4, scope: !24)
!31 = !DILocation(line: 5, scope: !32)
!32 = distinct !DILexicalBlock(scope: !4, file: !1, line: 5)
!33 = !DILocation(line: 5, scope: !4)
!34 = !DILocation(line: 6, scope: !32)
!35 = !DILocation(line: 7, scope: !4)
!36 = !DILocation(line: 5, scope: !32, inlinedAt: !37)
!37 = distinct !DILocation(line: 11, scope: !12)
!38 = !DILocation(line: 10, scope: !12)
!39 = !DILocation(line: 2, scope: !4, inlinedAt: !37)
!40 = !DILocation(line: 3, scope: !24, inlinedAt: !37)
!41 = !DILocation(line: 3, scope: !4, inlinedAt: !37)
!42 = !DILocation(line: 4, scope: !24, inlinedAt: !37)
!43 = !DILocation(line: 5, scope: !4, inlinedAt: !37)
!44 = !DILocation(line: 6, scope: !32, inlinedAt: !37)
!45 = !DILocation(line: 12, scope: !12)
