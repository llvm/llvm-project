; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump --name resource - | FileCheck %s
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT:  DW_AT_location	(DW_OP_reg1 W1)
; CHECK-NEXT:  DW_AT_abstract_origin {{.*}}"resource"
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


source_filename = "test.i"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

%struct.t = type { %struct.q* }
%struct.q = type { %struct.q*, i64 }

@tt = local_unnamed_addr global %struct.t* null, align 8, !dbg !0

; Function Attrs: noredzone nounwind readonly ssp
define i32 @g(%struct.t* nocapture readonly %t, i64 %r) local_unnamed_addr #0 !dbg !20 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.t* %t, metadata !26, metadata !DIExpression()), !dbg !29
  tail call void @llvm.dbg.value(metadata i64 %r, metadata !27, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.value(metadata %struct.t* %t, metadata !31, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata i64 %r, metadata !37, metadata !DIExpression()), !dbg !41
  %s.i5 = bitcast %struct.t* %t to %struct.q**
  tail call void @llvm.dbg.value(metadata %struct.q** %s.i5, metadata !38, metadata !DIExpression(DW_OP_deref)), !dbg !42
  %q.06.i = load %struct.q*, %struct.q** %s.i5, align 8
  tail call void @llvm.dbg.value(metadata %struct.q* %q.06.i, metadata !38, metadata !DIExpression()), !dbg !42
  %tobool7.i = icmp eq %struct.q* %q.06.i, null, !dbg !43
  br i1 %tobool7.i, label %find.exit, label %while.body.i.preheader, !dbg !43

while.body.i.preheader:                           ; preds = %entry
  br label %while.body.i, !dbg !44

while.body.i:                                     ; preds = %while.body.i.preheader, %if.end.i
  %q.08.i = phi %struct.q* [ %q.0.i, %if.end.i ], [ %q.06.i, %while.body.i.preheader ]
  %resource1.i = getelementptr inbounds %struct.q, %struct.q* %q.08.i, i64 0, i32 1, !dbg !44
  %0 = load i64, i64* %resource1.i, align 8, !dbg !44
  %cmp.i = icmp eq i64 %0, %r, !dbg !47
  br i1 %cmp.i, label %find.exit, label %if.end.i, !dbg !48

if.end.i:                                         ; preds = %while.body.i
  %next.i6 = bitcast %struct.q* %q.08.i to %struct.q**
  tail call void @llvm.dbg.value(metadata %struct.q** %next.i6, metadata !38, metadata !DIExpression(DW_OP_deref)), !dbg !42
  %q.0.i = load %struct.q*, %struct.q** %next.i6, align 8
  tail call void @llvm.dbg.value(metadata %struct.q* %q.0.i, metadata !38, metadata !DIExpression()), !dbg !42
  %tobool.i = icmp eq %struct.q* %q.0.i, null, !dbg !43
  br i1 %tobool.i, label %find.exit, label %while.body.i, !dbg !43, !llvm.loop !49

find.exit:                                        ; preds = %while.body.i, %if.end.i, %entry
  ret i32 undef, !dbg !52
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noredzone nounwind readonly ssp }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "tt", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (trunk 317516) (llvm/trunk 317518)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.i", directory: "/")
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
!19 = !{!"clang version 6.0.0 (trunk 317516) (llvm/trunk 317518)"}
!20 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 18, type: !21, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !25)
!21 = !DISubroutineType(types: !22)
!22 = !{!23, !24, !15}
!23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "t_t", file: !3, line: 1, baseType: !6)
!25 = !{!26, !27, !28}
!26 = !DILocalVariable(name: "t", arg: 1, scope: !20, file: !3, line: 18, type: !24)
!27 = !DILocalVariable(name: "r", arg: 2, scope: !20, file: !3, line: 18, type: !15)
!28 = !DILocalVariable(name: "q", scope: !20, file: !3, line: 19, type: !10)
!29 = !DILocation(line: 18, column: 11, scope: !20)
!30 = !DILocation(line: 18, column: 33, scope: !20)
!31 = !DILocalVariable(name: "t", arg: 1, scope: !32, file: !3, line: 9, type: !24)
!32 = distinct !DISubprogram(name: "find", scope: !3, file: !3, line: 9, type: !33, isLocal: true, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !36)
!33 = !DISubroutineType(types: !34)
!34 = !{!35, !24, !15}
!35 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!36 = !{!31, !37, !38}
!37 = !DILocalVariable(name: "resource", arg: 2, scope: !32, file: !3, line: 9, type: !15)
!38 = !DILocalVariable(name: "q", scope: !32, file: !3, line: 10, type: !10)
!39 = !DILocation(line: 9, column: 31, scope: !32, inlinedAt: !40)
!40 = distinct !DILocation(line: 20, column: 7, scope: !20)
!41 = !DILocation(line: 9, column: 53, scope: !32, inlinedAt: !40)
!42 = !DILocation(line: 10, column: 13, scope: !32, inlinedAt: !40)
!43 = !DILocation(line: 12, column: 3, scope: !32, inlinedAt: !40)
!44 = !DILocation(line: 13, column: 12, scope: !45, inlinedAt: !40)
!45 = distinct !DILexicalBlock(scope: !46, file: !3, line: 13, column: 9)
!46 = distinct !DILexicalBlock(scope: !32, file: !3, line: 12, column: 13)
!47 = !DILocation(line: 13, column: 21, scope: !45, inlinedAt: !40)
!48 = !DILocation(line: 13, column: 9, scope: !46, inlinedAt: !40)
!49 = distinct !{!49, !50, !51}
!50 = !DILocation(line: 12, column: 3, scope: !32)
!51 = !DILocation(line: 16, column: 3, scope: !32)
!52 = !DILocation(line: 24, column: 1, scope: !20)
