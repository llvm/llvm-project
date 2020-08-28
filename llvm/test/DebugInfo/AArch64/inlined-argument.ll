; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump --name resource - | FileCheck %s
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT:  DW_AT_location	(DW_OP_reg1 W1)
; CHECK-NEXT:  DW_AT_abstract_origin {{.*}}"resource"
;
; Inlined variable "resource"/!37 covers all blocks in its lexical scope. Check
; that it is given a single location.
;
; Generated from:
; typedef struct t *t_t;
; extern unsigned int enable;
; struct t {
;   struct q {
;     struct q *next;
;     unsigned long long resource;
;   } * s;
; } * tt;
; static unsigned long find(t_t t, unsigned long long resource) {
;   struct q *q;
;   q = t->s;
;   while (q) {
;     if (q->resource == resource)
;       return q;
;     q = q->next;
;   }
; }
; int g(t_t t, unsigned long long r) {
;   struct q *q;
;   q = find(t, r);
;   if (!q)
;     if (__builtin_expect(enable, 0)) {  }
; }

; ModuleID = 'inlined-arg.c'
source_filename = "inlined-arg.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

%struct.t = type { %struct.q* }
%struct.q = type { %struct.q*, i64 }

@tt = common local_unnamed_addr global %struct.t* null, align 8, !dbg !0

; Function Attrs: norecurse nounwind readonly ssp uwtable
define i32 @g(%struct.t* nocapture readonly %t, i64 %r) local_unnamed_addr !dbg !21 {
entry:
  call void @llvm.dbg.value(metadata %struct.t* %t, metadata !27, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i64 %r, metadata !28, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata %struct.t* %t, metadata !32, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i64 %r, metadata !37, metadata !DIExpression()), !dbg !41
  %s.i = getelementptr inbounds %struct.t, %struct.t* %t, i64 0, i32 0, !dbg !42
  %q.05.i = load %struct.q*, %struct.q** %s.i, align 8, !dbg !43, !tbaa !44
  call void @llvm.dbg.value(metadata %struct.q* %q.05.i, metadata !38, metadata !DIExpression()), !dbg !48
  %tobool6.i = icmp eq %struct.q* %q.05.i, null, !dbg !49
  br i1 %tobool6.i, label %find.exit, label %while.body.i, !dbg !49

while.body.i:                                     ; preds = %entry, %if.end.i
  %q.07.i = phi %struct.q* [ %q.0.i, %if.end.i ], [ %q.05.i, %entry ]
  %resource1.i = getelementptr inbounds %struct.q, %struct.q* %q.07.i, i64 0, i32 1, !dbg !50
  %0 = load i64, i64* %resource1.i, align 8, !dbg !50, !tbaa !53
  %cmp.i = icmp eq i64 %0, %r, !dbg !56
  br i1 %cmp.i, label %find.exit, label %if.end.i, !dbg !57

if.end.i:                                         ; preds = %while.body.i
  %next.i = getelementptr inbounds %struct.q, %struct.q* %q.07.i, i64 0, i32 0, !dbg !58
  %q.0.i = load %struct.q*, %struct.q** %next.i, align 8, !dbg !43, !tbaa !44
  call void @llvm.dbg.value(metadata %struct.q* %q.0.i, metadata !38, metadata !DIExpression()), !dbg !48
  %tobool.i = icmp eq %struct.q* %q.0.i, null, !dbg !49
  br i1 %tobool.i, label %find.exit, label %while.body.i, !dbg !49, !llvm.loop !59

find.exit:                                        ; preds = %while.body.i, %if.end.i, %entry
  call void @llvm.dbg.value(metadata %struct.q* undef, metadata !29, metadata !DIExpression()), !dbg !61
  ret i32 undef, !dbg !62
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17, !18, !19}
!llvm.ident = !{!20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "tt", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git cd3671d5dabc8848619d872f994770167a44ac5a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "inlined-arg.c", directory: "")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !3, line: 3, size: 64, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !7, file: !3, line: 7, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "q", file: !3, line: 4, size: 128, elements: !12)
!12 = !{!13, !14}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !11, file: !3, line: 5, baseType: !10, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "resource", scope: !11, file: !3, line: 6, baseType: !15, size: 64, offset: 64)
!15 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!16 = !{i32 2, !"Dwarf Version", i32 2}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{i32 7, !"PIC Level", i32 2}
!20 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git cd3671d5dabc8848619d872f994770167a44ac5a)"}
!21 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 19, type: !22, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !26)
!22 = !DISubroutineType(types: !23)
!23 = !{!24, !25, !15}
!24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "t_t", file: !3, line: 1, baseType: !6)
!26 = !{!27, !28, !29}
!27 = !DILocalVariable(name: "t", arg: 1, scope: !21, file: !3, line: 19, type: !25)
!28 = !DILocalVariable(name: "r", arg: 2, scope: !21, file: !3, line: 19, type: !15)
!29 = !DILocalVariable(name: "q", scope: !21, file: !3, line: 20, type: !10)
!30 = !DILocation(line: 19, column: 11, scope: !21)
!31 = !DILocation(line: 19, column: 33, scope: !21)
!32 = !DILocalVariable(name: "t", arg: 1, scope: !33, file: !3, line: 10, type: !25)
!33 = distinct !DISubprogram(name: "find", scope: !3, file: !3, line: 10, type: !34, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !36)
!34 = !DISubroutineType(types: !35)
!35 = !{!10, !25, !15}
!36 = !{!32, !37, !38}
!37 = !DILocalVariable(name: "resource", arg: 2, scope: !33, file: !3, line: 10, type: !15)
!38 = !DILocalVariable(name: "q", scope: !33, file: !3, line: 11, type: !10)
!39 = !DILocation(line: 10, column: 27, scope: !33, inlinedAt: !40)
!40 = distinct !DILocation(line: 21, column: 7, scope: !21)
!41 = !DILocation(line: 10, column: 49, scope: !33, inlinedAt: !40)
!42 = !DILocation(line: 12, column: 10, scope: !33, inlinedAt: !40)
!43 = !DILocation(line: 0, scope: !33, inlinedAt: !40)
!44 = !{!45, !45, i64 0}
!45 = !{!"any pointer", !46, i64 0}
!46 = !{!"omnipotent char", !47, i64 0}
!47 = !{!"Simple C/C++ TBAA"}
!48 = !DILocation(line: 11, column: 13, scope: !33, inlinedAt: !40)
!49 = !DILocation(line: 13, column: 3, scope: !33, inlinedAt: !40)
!50 = !DILocation(line: 14, column: 12, scope: !51, inlinedAt: !40)
!51 = distinct !DILexicalBlock(scope: !52, file: !3, line: 14, column: 9)
!52 = distinct !DILexicalBlock(scope: !33, file: !3, line: 13, column: 13)
!53 = !{!54, !55, i64 8}
!54 = !{!"q", !45, i64 0, !55, i64 8}
!55 = !{!"long long", !46, i64 0}
!56 = !DILocation(line: 14, column: 21, scope: !51, inlinedAt: !40)
!57 = !DILocation(line: 14, column: 9, scope: !52, inlinedAt: !40)
!58 = !DILocation(line: 16, column: 12, scope: !52, inlinedAt: !40)
!59 = distinct !{!59, !49, !60}
!60 = !DILocation(line: 17, column: 3, scope: !33, inlinedAt: !40)
!61 = !DILocation(line: 20, column: 13, scope: !21)
!62 = !DILocation(line: 24, column: 1, scope: !21)
