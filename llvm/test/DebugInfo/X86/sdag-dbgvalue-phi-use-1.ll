; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE
; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF

; This test case was generated from the following debug.c program,
; using: clang debug.c -g -O1 -S -o dbg_value_phi_isel1.ll -emit-llvm
; --------------------------------------
; int end = 10;
;
; int main() {
;   int x = 9;
;   int y = 13;
;   for (int u = 0; u < end; ++u) {
;     x += y;
;     y = y * 3;
;   }
;
;   volatile int arr[80];
;   for (int q = 0; q < 64; ++q) {
;     arr[q] = q + 3;
;   }
;
;   return x;
; }
; --------------------------------------
;

; ModuleID = 'debug.c'
source_filename = "debug.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@end = dso_local local_unnamed_addr global i32 10, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !11 {
; CHECK-LABEL: name:            main
entry:
  %arr = alloca [80 x i32], align 16
  call void @llvm.dbg.value(metadata i32 9, metadata !15, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i32 13, metadata !16, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !28
  %0 = load i32, ptr @end, align 4, !dbg !29, !tbaa !31
  %cmp20 = icmp sgt i32 %0, 0, !dbg !35
  br i1 %cmp20, label %for.body.lr.ph, label %for.cond.cleanup, !dbg !36

for.body.lr.ph:                                   ; preds = %entry
  %1 = load i32, ptr @end, align 4, !tbaa !31
  br label %for.body, !dbg !36

for.cond.cleanup:                                 ; preds = %for.body, %entry
; CHECK-LABEL: bb.{{.*}}.for.cond.cleanup:
; CHECK:         [[REG1:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 7
; INSTRREF-NEXT: DBG_INSTR_REF 7, 0
; DBGVALUE-NEXT: DBG_VALUE [[REG1]]
  %x.0.lcssa = phi i32 [ 9, %entry ], [ %add, %for.body ]
  call void @llvm.dbg.value(metadata i32 %x.0.lcssa, metadata !15, metadata !DIExpression()), !dbg !26
  call void @llvm.lifetime.start.p0(i64 320, ptr nonnull %arr) #3, !dbg !37
  call void @llvm.dbg.declare(metadata ptr %arr, metadata !19, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !39
  br label %for.body4, !dbg !40

for.body:                                         ; preds = %for.body.lr.ph, %for.body
; CHECK-LABEL: bb.{{.*}}.for.body:
; CHECK:        [[REG2:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:   debug-instr-number 3
; CHECK-NEXT:   [[REG3:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:   debug-instr-number 4
; CHECK-NEXT:   [[REG4:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:   debug-instr-number 5
; INSTRREF-NEXT: DBG_INSTR_REF 3, 0
; INSTRREF-NEXT: DBG_INSTR_REF 4, 0
; INSTRREF-NEXT: DBG_INSTR_REF 5, 0
; DBGVALUE-NEXT: DBG_VALUE [[REG2]]
; DBGVALUE-NEXT: DBG_VALUE [[REG3]]
; DBGVALUE-NEXT: DBG_VALUE [[REG4]]
  %u.023 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %y.022 = phi i32 [ 13, %for.body.lr.ph ], [ %mul, %for.body ]
  %x.021 = phi i32 [ 9, %for.body.lr.ph ], [ %add, %for.body ]
  call void @llvm.dbg.value(metadata i32 %u.023, metadata !17, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %y.022, metadata !16, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %x.021, metadata !15, metadata !DIExpression()), !dbg !26
  %add = add nuw nsw i32 %y.022, %x.021, !dbg !41
  %mul = mul nsw i32 %y.022, 3, !dbg !43
  %inc = add nuw nsw i32 %u.023, 1, !dbg !44
  call void @llvm.dbg.value(metadata i32 %inc, metadata !17, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i32 %mul, metadata !16, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %add, metadata !15, metadata !DIExpression()), !dbg !26
  %cmp = icmp slt i32 %inc, %1, !dbg !35
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !36, !llvm.loop !45

for.cond.cleanup3:                                ; preds = %for.body4
  call void @llvm.lifetime.end.p0(i64 320, ptr nonnull %arr) #3, !dbg !47
  ret i32 %x.0.lcssa, !dbg !48

for.body4:                                        ; preds = %for.body4, %for.cond.cleanup
  %indvars.iv = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next, %for.body4 ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !24, metadata !DIExpression()), !dbg !39
  %arrayidx = getelementptr inbounds [80 x i32], ptr %arr, i64 0, i64 %indvars.iv, !dbg !49
  %2 = trunc i64 %indvars.iv to i32, !dbg !52
  %3 = add i32 %2, 3, !dbg !52
  store volatile i32 %3, ptr %arrayidx, align 4, !dbg !52, !tbaa !31
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !53
  call void @llvm.dbg.value(metadata i32 undef, metadata !24, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !39
  %exitcond = icmp eq i64 %indvars.iv.next, 64, !dbg !54
  br i1 %exitcond, label %for.cond.cleanup3, label %for.body4, !dbg !40, !llvm.loop !55
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "end", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (x)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "debug.c", directory: "")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 7.0.0 (x)"}
!11 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !{!15, !16, !17, !19, !24}
!15 = !DILocalVariable(name: "x", scope: !11, file: !3, line: 4, type: !6)
!16 = !DILocalVariable(name: "y", scope: !11, file: !3, line: 5, type: !6)
!17 = !DILocalVariable(name: "u", scope: !18, file: !3, line: 6, type: !6)
!18 = distinct !DILexicalBlock(scope: !11, file: !3, line: 6, column: 3)
!19 = !DILocalVariable(name: "arr", scope: !11, file: !3, line: 11, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 2560, elements: !22)
!21 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!22 = !{!23}
!23 = !DISubrange(count: 80)
!24 = !DILocalVariable(name: "q", scope: !25, file: !3, line: 12, type: !6)
!25 = distinct !DILexicalBlock(scope: !11, file: !3, line: 12, column: 3)
!26 = !DILocation(line: 4, column: 7, scope: !11)
!27 = !DILocation(line: 5, column: 7, scope: !11)
!28 = !DILocation(line: 6, column: 12, scope: !18)
!29 = !DILocation(line: 6, column: 23, scope: !30)
!30 = distinct !DILexicalBlock(scope: !18, file: !3, line: 6, column: 3)
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !DILocation(line: 6, column: 21, scope: !30)
!36 = !DILocation(line: 6, column: 3, scope: !18)
!37 = !DILocation(line: 11, column: 3, scope: !11)
!38 = !DILocation(line: 11, column: 16, scope: !11)
!39 = !DILocation(line: 12, column: 12, scope: !25)
!40 = !DILocation(line: 12, column: 3, scope: !25)
!41 = !DILocation(line: 7, column: 7, scope: !42)
!42 = distinct !DILexicalBlock(scope: !30, file: !3, line: 6, column: 33)
!43 = !DILocation(line: 8, column: 11, scope: !42)
!44 = !DILocation(line: 6, column: 28, scope: !30)
!45 = distinct !{!45, !36, !46}
!46 = !DILocation(line: 9, column: 3, scope: !18)
!47 = !DILocation(line: 17, column: 1, scope: !11)
!48 = !DILocation(line: 16, column: 3, scope: !11)
!49 = !DILocation(line: 13, column: 5, scope: !50)
!50 = distinct !DILexicalBlock(scope: !51, file: !3, line: 12, column: 32)
!51 = distinct !DILexicalBlock(scope: !25, file: !3, line: 12, column: 3)
!52 = !DILocation(line: 13, column: 12, scope: !50)
!53 = !DILocation(line: 12, column: 27, scope: !51)
!54 = !DILocation(line: 12, column: 21, scope: !51)
!55 = distinct !{!55, !40, !56}
!56 = !DILocation(line: 14, column: 3, scope: !25)
