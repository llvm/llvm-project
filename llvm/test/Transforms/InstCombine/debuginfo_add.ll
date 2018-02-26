; RUN: opt -instcombine %s -o - -S | FileCheck %s
; typedef struct v *v_t;
; struct v {
;   unsigned long long p;
; };
;  
; void f(v_t object, unsigned long long *start) {
;   unsigned head_size;
;   unsigned long long orig_start;
;   unsigned long long offset;
;   orig_start = *start;
;   for (offset = orig_start - (unsigned long long)(1 << 12); head_size;
;        offset -= (unsigned long long)(1 << 12), head_size -= (1 << 12))
;     use(offset, (object));
; }
source_filename = "test.i"
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios5.0.0"

%struct.vm_object = type { i64 }

; Function Attrs: nounwind ssp
define void @f(%struct.vm_object* %object, i64* nocapture readonly %start) local_unnamed_addr #0 !dbg !11 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.vm_object* %object, metadata !21, metadata !DIExpression()), !dbg !27
  tail call void @llvm.dbg.value(metadata i64* %start, metadata !22, metadata !DIExpression()), !dbg !28
  %0 = load i64, i64* %start, align 4, !dbg !29
  tail call void @llvm.dbg.value(metadata i64 %0, metadata !25, metadata !DIExpression()), !dbg !30
  %offset.08 = add i64 %0, -4096
  tail call void @llvm.dbg.value(metadata i64 %offset.08, metadata !26, metadata !DIExpression()), !dbg !31
  ; CHECK: call void @llvm.dbg.value(metadata i64 %0, metadata !26, metadata !DIExpression(DW_OP_constu, 4096, DW_OP_minus, DW_OP_stack_value)), !dbg !30
  tail call void @llvm.dbg.value(metadata i32 undef, metadata !23, metadata !DIExpression()), !dbg !32
  br i1 undef, label %for.end, label %for.body.lr.ph, !dbg !32

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !32

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %offset.010 = phi i64 [ %offset.08, %for.body.lr.ph ], [ %offset.0, %for.body ]
  %head_size.09 = phi i32 [ undef, %for.body.lr.ph ], [ %sub2, %for.body ]
  tail call void @llvm.dbg.value(metadata i32 %head_size.09, metadata !23, metadata !DIExpression()), !dbg !31
  %call = tail call i32 bitcast (i32 (...)* @use to i32 (i64, %struct.vm_object*)*)(i64 %offset.010, %struct.vm_object* %object) #3, !dbg !34
  %sub2 = add i32 %head_size.09, -4096, !dbg !37
  %offset.0 = add i64 %offset.010, -4096
  tail call void @llvm.dbg.value(metadata i64 %offset.0, metadata !26, metadata !DIExpression()), !dbg !30
  ; CHECK: call void @llvm.dbg.value(metadata i64 %offset.010, metadata !26, metadata !DIExpression(DW_OP_constu, 4096, DW_OP_minus, DW_OP_stack_value)), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %sub2, metadata !23, metadata !DIExpression()), !dbg !31
  %tobool = icmp eq i32 %sub2, 0, !dbg !32
  br i1 %tobool, label %for.end, label %for.body, !dbg !32, !llvm.loop !38

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !40
}

declare i32 @use(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nobuiltin }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 317434) (llvm/trunk 317437)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "test.i", directory: "/Data/radar/31209283")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!5 = !{i32 2, !"Dwarf Version", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 1, !"min_enum_size", i32 4}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{!"clang version 6.0.0 (trunk 317434) (llvm/trunk 317437)"}
!11 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 6, type: !12, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !20)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !19}
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "v_t", file: !1, line: 1, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 32)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "v", file: !1, line: 2, size: 64, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !16, file: !1, line: 3, baseType: !4, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 32)
!20 = !{!21, !22, !23, !25, !26}
!21 = !DILocalVariable(name: "object", arg: 1, scope: !11, file: !1, line: 6, type: !14)
!22 = !DILocalVariable(name: "start", arg: 2, scope: !11, file: !1, line: 6, type: !19)
!23 = !DILocalVariable(name: "head_size", scope: !11, file: !1, line: 7, type: !24)
!24 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!25 = !DILocalVariable(name: "orig_start", scope: !11, file: !1, line: 8, type: !4)
!26 = !DILocalVariable(name: "offset", scope: !11, file: !1, line: 9, type: !4)
!27 = !DILocation(line: 6, column: 20, scope: !11)
!28 = !DILocation(line: 6, column: 48, scope: !11)
!29 = !DILocation(line: 8, column: 22, scope: !11)
!30 = !DILocation(line: 7, column: 12, scope: !11)
!31 = !DILocation(line: 10, column: 16, scope: !11)
!32 = !DILocation(line: 11, column: 5, scope: !33)
!33 = distinct !DILexicalBlock(scope: !11, file: !1, line: 11, column: 5)
!34 = !DILocation(line: 13, column: 7, scope: !35)
!35 = distinct !DILexicalBlock(scope: !36, file: !1, line: 12, column: 75)
!36 = distinct !DILexicalBlock(scope: !33, file: !1, line: 11, column: 5)
!37 = !DILocation(line: 12, column: 61, scope: !36)
!38 = distinct !{!38, !32, !39}
!39 = !DILocation(line: 14, column: 3, scope: !33)
!40 = !DILocation(line: 15, column: 1, scope: !11)
