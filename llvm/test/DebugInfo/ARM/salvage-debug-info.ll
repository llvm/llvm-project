; RUN: opt -codegenprepare -S %s -o - | FileCheck %s
; typedef struct info {
;   unsigned long long size;
; } info_t;
; extern unsigned p;
; extern unsigned n;
; void f() {
;   unsigned int i;
;   if (p) {
;     info_t *info = (info_t *)p;
;     for (i = 0; i < n; i++)
;       use(info[i].size);
;   }
; }
source_filename = "debug.i"
target datalayout = "e-m:o-p:32:32-i64:64-a:0:32-n32-S128"
target triple = "thumbv7k-apple-ios10.0.0"

%struct.info = type { i64 }

@p = external local_unnamed_addr global i32, align 4
@n = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind ssp uwtable
define void @f() local_unnamed_addr #0 !dbg !16 {
entry:
  %0 = load i32, i32* @p, align 4, !dbg !25
  %tobool = icmp eq i32 %0, 0, !dbg !25
  br i1 %tobool, label %if.end, label %if.then, !dbg !26

if.then:                                          ; preds = %entry
  %1 = inttoptr i32 %0 to %struct.info*, !dbg !27
  tail call void @llvm.dbg.value(metadata %struct.info* %1, metadata !22, metadata !DIExpression()), !dbg !28
  ; CHECK: call void @llvm.dbg.value(metadata i32 %0, metadata !22, metadata !DIExpression())
  tail call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression()), !dbg !29
  %2 = load i32, i32* @n, align 4, !dbg !30
  %cmp5 = icmp eq i32 %2, 0, !dbg !33
  br i1 %cmp5, label %if.end, label %for.body.preheader, !dbg !34

for.body.preheader:                               ; preds = %if.then
  ; CHECK: for.body.preheader:
  ; CHECK:   %2 = inttoptr i32 %0 to %struct.info*
  br label %for.body, !dbg !35

for.body:                                         ; preds = %for.body.preheader, %for.body
  %lsr.iv = phi %struct.info* [ %1, %for.body.preheader ], [ %scevgep, %for.body ]
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %lsr.iv7 = bitcast %struct.info* %lsr.iv to i64*
  tail call void @llvm.dbg.value(metadata i32 %i.06, metadata !20, metadata !DIExpression()), !dbg !29
  %3 = load i64, i64* %lsr.iv7, align 8, !dbg !35
  %call = tail call i32 bitcast (i32 (...)* @use to i32 (i64)*)(i64 %3) #3, !dbg !36
  %inc = add nuw i32 %i.06, 1, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !20, metadata !DIExpression()), !dbg !29
  %4 = load i32, i32* @n, align 4, !dbg !30
  %scevgep = getelementptr %struct.info, %struct.info* %lsr.iv, i32 1, !dbg !33
  %cmp = icmp ult i32 %inc, %4, !dbg !33
  br i1 %cmp, label %for.body, label %if.end.loopexit, !dbg !34, !llvm.loop !38

if.end.loopexit:                                  ; preds = %for.body
  br label %if.end, !dbg !40

if.end:                                           ; preds = %if.end.loopexit, %if.then, %entry
  ret void, !dbg !40
}
declare i32 @use(...) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nobuiltin nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 317231) (llvm/trunk 317262)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "debug.i", directory: "/Data/radar/35321562")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 32)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "info_t", file: !1, line: 3, baseType: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "info", file: !1, line: 1, size: 64, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "size", scope: !6, file: !1, line: 2, baseType: !9, size: 64)
!9 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 1, !"min_enum_size", i32 4}
!14 = !{i32 7, !"PIC Level", i32 2}
!15 = !{!"clang version 6.0.0 (trunk 317231) (llvm/trunk 317262)"}
!16 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 6, type: !17, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !0, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !{!20, !22}
!20 = !DILocalVariable(name: "i", scope: !16, file: !1, line: 7, type: !21)
!21 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!22 = !DILocalVariable(name: "info", scope: !23, file: !1, line: 9, type: !4)
!23 = distinct !DILexicalBlock(scope: !24, file: !1, line: 8, column: 10)
!24 = distinct !DILexicalBlock(scope: !16, file: !1, line: 8, column: 7)
!25 = !DILocation(line: 8, column: 7, scope: !24)
!26 = !DILocation(line: 8, column: 7, scope: !16)
!27 = !DILocation(line: 9, column: 20, scope: !23)
!28 = !DILocation(line: 9, column: 13, scope: !23)
!29 = !DILocation(line: 7, column: 16, scope: !16)
!30 = !DILocation(line: 10, column: 21, scope: !31)
!31 = distinct !DILexicalBlock(scope: !32, file: !1, line: 10, column: 5)
!32 = distinct !DILexicalBlock(scope: !23, file: !1, line: 10, column: 5)
!33 = !DILocation(line: 10, column: 19, scope: !31)
!34 = !DILocation(line: 10, column: 5, scope: !32)
!35 = !DILocation(line: 11, column: 19, scope: !31)
!36 = !DILocation(line: 11, column: 7, scope: !31)
!37 = !DILocation(line: 10, column: 25, scope: !31)
!38 = distinct !{!38, !34, !39}
!39 = !DILocation(line: 11, column: 23, scope: !32)
!40 = !DILocation(line: 13, column: 1, scope: !16)
