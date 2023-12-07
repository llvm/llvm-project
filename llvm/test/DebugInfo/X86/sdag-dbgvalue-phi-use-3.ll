; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:     -experimental-debug-variable-locations=false  \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE
; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:     -experimental-debug-variable-locations=true  \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF

; This test case was generated from the following phi-split.c program,
; using: clang phi-split.c -g -O1 -S -o - --target=i386 -emit-llvm
; --------------------------------------
; long long end = 10;
;
; int main() {
;   long long x = 9;
;   long long y = 13;
;   for (long long u = 0; u < end; ++u) {
;     x += y;
;     y = y * 3;
;   }
;
;   volatile long long arr[80];
;   for (long long q = 0; q < 64; ++q) {
;     arr[q] = q + 3;
;   }
;
;   return x;
; }
; --------------------------------------
;

; ModuleID = 'phi-split.c'
source_filename = "phi-split.c"
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386"

@end = dso_local local_unnamed_addr global i64 10, align 8, !dbg !0

; Function Attrs: nounwind
define dso_local i32 @main() local_unnamed_addr #0 !dbg !12 {
; CHECK-LABEL: name:            main
entry:
  %arr = alloca [80 x i64], align 8
  call void @llvm.dbg.value(metadata i64 9, metadata !17, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i64 13, metadata !18, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !30
  %0 = load i64, ptr @end, align 8, !dbg !31
  %cmp20 = icmp sgt i64 %0, 0, !dbg !37
  br i1 %cmp20, label %for.body.lr.ph, label %for.cond.cleanup, !dbg !38

for.body.lr.ph:                                   ; preds = %entry
  %1 = load i64, ptr @end, align 8
  br label %for.body, !dbg !38

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %extract.t = trunc i64 %add to i32, !dbg !38
  br label %for.cond.cleanup, !dbg !39

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %x.0.lcssa.off0 = phi i32 [ 9, %entry ], [ %extract.t, %for.cond.cleanup.loopexit ]
  call void @llvm.dbg.value(metadata i64 undef, metadata !17, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.value(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !41
  br label %for.body4, !dbg !42

for.body:                                         ; preds = %for.body.lr.ph, %for.body
; CHECK-LABEL: bb.{{.*}}.for.body:
; CHECK:         [[REG2:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 5
; CHECK-NEXT:    [[REG3:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 6
; CHECK-NEXT:    [[REG4:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 7
; CHECK-NEXT:    [[REG5:%[0-9]+]]:gr32_nosp = PHI
; INSTRREF-SAME:    debug-instr-number 8
; CHECK-NEXT:    [[REG6:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 9
; CHECK-NEXT:    [[REG7:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 10
; INSTRREF-NEXT: DBG_INSTR_REF !19, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 0, 32), dbg-instr-ref(5, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !19, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 32, 32), dbg-instr-ref(6, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !18, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 0, 32), dbg-instr-ref(7, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !18, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 32, 32), dbg-instr-ref(8, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !17, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 0, 32), dbg-instr-ref(9, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !17, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 32, 32), dbg-instr-ref(10, 0)
; DBGVALUE-NEXT: DBG_VALUE [[REG2]], $noreg, !19, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG3]], $noreg, !19, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG4]], $noreg, !18, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG5]], $noreg, !18, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG6]], $noreg, !17, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG7]], $noreg, !17, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
  %u.023 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %y.022 = phi i64 [ 13, %for.body.lr.ph ], [ %mul, %for.body ]
  %x.021 = phi i64 [ 9, %for.body.lr.ph ], [ %add, %for.body ]
  call void @llvm.dbg.value(metadata i64 %u.023, metadata !19, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i64 %y.022, metadata !18, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i64 %x.021, metadata !17, metadata !DIExpression()), !dbg !28
  %add = add nuw nsw i64 %y.022, %x.021, !dbg !43
  %mul = mul nsw i64 %y.022, 3, !dbg !45
  %inc = add nuw nsw i64 %u.023, 1, !dbg !46
  call void @llvm.dbg.value(metadata i64 %inc, metadata !19, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i64 %mul, metadata !18, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata i64 %add, metadata !17, metadata !DIExpression()), !dbg !28
  %cmp = icmp slt i64 %inc, %1, !dbg !37
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !dbg !38, !llvm.loop !47

for.cond.cleanup3:                                ; preds = %for.body4
  ret i32 %x.0.lcssa.off0, !dbg !50

for.body4:                                        ; preds = %for.cond.cleanup, %for.body4
  %q.019 = phi i64 [ 0, %for.cond.cleanup ], [ %inc7, %for.body4 ]
  call void @llvm.dbg.value(metadata i64 %q.019, metadata !26, metadata !DIExpression()), !dbg !41
  %add5 = add nuw nsw i64 %q.019, 3, !dbg !51
  %idxprom = trunc i64 %q.019 to i32, !dbg !54
  %arrayidx = getelementptr inbounds [80 x i64], ptr %arr, i32 0, i32 %idxprom, !dbg !54
  store volatile i64 %add5, ptr %arrayidx, align 8, !dbg !55
  %inc7 = add nuw nsw i64 %q.019, 1, !dbg !56
  call void @llvm.dbg.value(metadata i64 %inc7, metadata !26, metadata !DIExpression()), !dbg !41
  %cmp2 = icmp ult i64 %inc7, 64, !dbg !57
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3, !dbg !42, !llvm.loop !58
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "end", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0  (x)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "phi-split.c", directory: "")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!7 = !{i32 1, !"NumRegisterParameters", i32 0}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 7.0.0  (x)"}
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 3, type: !13, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !2, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17, !18, !19, !21, !26}
!17 = !DILocalVariable(name: "x", scope: !12, file: !3, line: 4, type: !6)
!18 = !DILocalVariable(name: "y", scope: !12, file: !3, line: 5, type: !6)
!19 = !DILocalVariable(name: "u", scope: !20, file: !3, line: 6, type: !6)
!20 = distinct !DILexicalBlock(scope: !12, file: !3, line: 6, column: 3)
!21 = !DILocalVariable(name: "arr", scope: !12, file: !3, line: 11, type: !22)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !23, size: 5120, elements: !24)
!23 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!24 = !{!25}
!25 = !DISubrange(count: 80)
!26 = !DILocalVariable(name: "q", scope: !27, file: !3, line: 12, type: !6)
!27 = distinct !DILexicalBlock(scope: !12, file: !3, line: 12, column: 3)
!28 = !DILocation(line: 4, column: 13, scope: !12)
!29 = !DILocation(line: 5, column: 13, scope: !12)
!30 = !DILocation(line: 6, column: 18, scope: !20)
!31 = !DILocation(line: 6, column: 29, scope: !32)
!32 = distinct !DILexicalBlock(scope: !20, file: !3, line: 6, column: 3)
!37 = !DILocation(line: 6, column: 27, scope: !32)
!38 = !DILocation(line: 6, column: 3, scope: !20)
!39 = !DILocation(line: 11, column: 3, scope: !12)
!40 = !DILocation(line: 11, column: 22, scope: !12)
!41 = !DILocation(line: 12, column: 18, scope: !27)
!42 = !DILocation(line: 12, column: 3, scope: !27)
!43 = !DILocation(line: 7, column: 7, scope: !44)
!44 = distinct !DILexicalBlock(scope: !32, file: !3, line: 6, column: 39)
!45 = !DILocation(line: 8, column: 11, scope: !44)
!46 = !DILocation(line: 6, column: 34, scope: !32)
!47 = distinct !{!47, !38, !48}
!48 = !DILocation(line: 9, column: 3, scope: !20)
!50 = !DILocation(line: 16, column: 3, scope: !12)
!51 = !DILocation(line: 13, column: 16, scope: !52)
!52 = distinct !DILexicalBlock(scope: !53, file: !3, line: 12, column: 38)
!53 = distinct !DILexicalBlock(scope: !27, file: !3, line: 12, column: 3)
!54 = !DILocation(line: 13, column: 5, scope: !52)
!55 = !DILocation(line: 13, column: 12, scope: !52)
!56 = !DILocation(line: 12, column: 33, scope: !53)
!57 = !DILocation(line: 12, column: 27, scope: !53)
!58 = distinct !{!58, !42, !59}
!59 = !DILocation(line: 14, column: 3, scope: !27)
