; REQUIRES: asserts
; RUN: llvm-profdata merge --sample -profile-isfs --extbinary -o %t.afdo %S/Inputs/fsloader-probe.afdo
; RUN: llc -enable-fs-discriminator -fs-profile-file=%t.afdo -show-fs-branchprob -disable-ra-fsprofile-loader=false -disable-layout-fsprofile-loader=false < %s 2>&1 | FileCheck %s --check-prefix=LOADER
;
;;
;; C source code for the test.
;; Compiled with clang -O3 -g -fdebug-info-for-profiling -fpseudo-probe-for-profiling -mllvm --enable-fs-discriminator
;; // A test case for loop unroll.
;;
;; __attribute__((noinline)) int bar(int i){
;;   volatile int j;
;;   j = i;
;;   return j;
;; }
;;
;; unsigned sum;
;; __attribute__((noinline)) void work(int i){
;;   if (sum % 7)
;;     sum += i;
;;   else
;;     sum -= i;
;; }
;;
;; __attribute__((noinline)) void foo(){
;;   int i, j;
;;   for (j = 0; j < 48; j++)
;;     for (i = 0; i < 4; i++) {
;;       int ii = bar(i+j*48);
;;       if (ii % 2)
;;         work(ii*2);
;;       if (ii % 4)
;;         work(ii*3);
;;   }
;; }
;;
;; int main() {
;;   int i;
;;   for (i = 0; i < 10000000; i++) {
;;     foo();
;;   }
;; }
;;
;;

;; Check that new branch probs are generated.

; LOADER: Set branch fs prob: MBB (3 -> 5): unroll.c:22:12-->unroll.c:20:12 W=44114  0x30000000 / 0x80000000 = 37.50% --> 0x80000000 / 0x80000000 = 100.00%
; LOADER: Set branch fs prob: MBB (3 -> 4): unroll.c:22:12 W=44114  0x50000000 / 0x80000000 = 62.50% --> 0x00000000 / 0x80000000 = 0.00%
; LOADER: Set branch fs prob: MBB (9 -> 11): unroll.c:20:12-->unroll.c:22:12 W=44114  0x40000000 / 0x80000000 = 50.00% --> 0x80000000 / 0x80000000 = 100.00%
; LOADER: Set branch fs prob: MBB (9 -> 10): unroll.c:20:12 W=44114  0x40000000 / 0x80000000 = 50.00% --> 0x00000000 / 0x80000000 = 0.00%
; LOADER: Set branch fs prob: MBB (1 -> 3): unroll.c:20:12-->unroll.c:22:12 W=26128  0x34de9bd3 / 0x80000000 = 41.30% --> 0x80000000 / 0x80000000 = 100.00%
; LOADER: Set branch fs prob: MBB (1 -> 2): unroll.c:20:12 W=26128  0x4b21642d / 0x80000000 = 58.70% --> 0x00000000 / 0x80000000 = 0.00%
; LOADER: Set branch fs prob: MBB (5 -> 7): unroll.c:20:12-->unroll.c:22:12 W=26128  0x34693ef1 / 0x80000000 = 40.95% --> 0x0060917b / 0x80000000 = 0.29%
; LOADER: Set branch fs prob: MBB (5 -> 6): unroll.c:20:12 W=26128  0x4b96c10f / 0x80000000 = 59.05% --> 0x7f9f6e85 / 0x80000000 = 99.71%
; LOADER: Set branch fs prob: MBB (7 -> 9): unroll.c:22:12-->unroll.c:20:12 W=26128  0x34300cd0 / 0x80000000 = 40.77% --> 0x00000000 / 0x80000000 = 0.00%
; LOADER: Set branch fs prob: MBB (7 -> 8): unroll.c:22:12 W=26128  0x4bcff330 / 0x80000000 = 59.23% --> 0x80000000 / 0x80000000 = 100.00%
; LOADER: Set branch fs prob: MBB (11 -> 13): unroll.c:22:12-->unroll.c:20:12 W=26128  0x35c65cf7 / 0x80000000 = 42.01% --> 0x02ae02d2 / 0x80000000 = 2.09%
; LOADER: Set branch fs prob: MBB (11 -> 12): unroll.c:22:12 W=26128  0x4a39a309 / 0x80000000 = 57.99% --> 0x7d51fd2e / 0x80000000 = 97.91%
; LOADER: Set branch fs prob: MBB (13 -> 15): unroll.c:20:12-->unroll.c:22:12 W=26128  0x34de9bd3 / 0x80000000 = 41.30% --> 0x0126b8ac / 0x80000000 = 0.90%
; LOADER: Set branch fs prob: MBB (13 -> 14): unroll.c:20:12 W=26128  0x4b21642d / 0x80000000 = 58.70% --> 0x7ed94754 / 0x80000000 = 99.10%
; LOADER: Set branch fs prob: MBB (15 -> 17): unroll.c:22:12-->unroll.c:17:4 W=26128  0x3949278b / 0x80000000 = 44.75% --> 0x089b8337 / 0x80000000 = 6.72%
; LOADER: Set branch fs prob: MBB (15 -> 16): unroll.c:22:12 W=26128  0x46b6d875 / 0x80000000 = 55.25% --> 0x77647cc9 / 0x80000000 = 93.28%



target triple = "x86_64-unknown-linux-gnu"


@sum = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0
@__llvm_fs_discriminator__ = weak_odr constant i1 true
@llvm.used = appending global [1 x ptr] [ptr @__llvm_fs_discriminator__], section "llvm.metadata"

; Function Attrs: nofree noinline nounwind memory(inaccessiblemem: readwrite) uwtable
declare dso_local i32 @bar(i32 noundef %i) local_unnamed_addr #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
declare dso_local void @work(i32 noundef %i) local_unnamed_addr #3

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo() local_unnamed_addr #4 !dbg !47 {
entry:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !59
  call void @llvm.dbg.value(metadata i32 0, metadata !52, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !61
  br label %for.cond1.preheader, !dbg !63

for.cond1.preheader:                              ; preds = %entry, %if.end9.3
  %lsr.iv = phi i32 [ 3, %entry ], [ %lsr.iv.next, %if.end9.3 ]
  call void @llvm.dbg.value(metadata i32 %lsr.iv, metadata !52, metadata !DIExpression(DW_OP_consts, 3, DW_OP_minus, DW_OP_consts, 48, DW_OP_div, DW_OP_stack_value)), !dbg !60
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !65
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 5, i32 0, i64 -1), !dbg !67
  %0 = add i32 %lsr.iv, -3, !dbg !65
  call void @llvm.dbg.value(metadata i32 0, metadata !51, metadata !DIExpression()), !dbg !60
  %call = tail call i32 @bar(i32 noundef %0), !dbg !68
  call void @llvm.dbg.value(metadata i32 %call, metadata !53, metadata !DIExpression()), !dbg !70
  %1 = and i32 %call, 1, !dbg !71
  %tobool.not = icmp eq i32 %1, 0, !dbg !71
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !73

if.then:                                          ; preds = %for.cond1.preheader
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1), !dbg !74
  %mul4 = shl nsw i32 %call, 1, !dbg !75
  tail call void @work(i32 noundef %mul4), !dbg !76
  br label %if.end, !dbg !78

if.end:                                           ; preds = %if.then, %for.cond1.preheader
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 7, i32 0, i64 -1), !dbg !79
  %2 = and i32 %call, 3, !dbg !81
  %tobool6.not = icmp eq i32 %2, 0, !dbg !81
  br i1 %tobool6.not, label %if.end9, label %if.then7, !dbg !82

if.then7:                                         ; preds = %if.end
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 8, i32 0, i64 -1), !dbg !83
  %mul8 = mul nsw i32 %call, 3, !dbg !84
  tail call void @work(i32 noundef %mul8), !dbg !85
  br label %if.end9, !dbg !87

if.end9:                                          ; preds = %if.then7, %if.end
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 9, i32 0, i64 -1), !dbg !88
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 10, i32 0, i64 -1), !dbg !89
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !65
  call void @llvm.dbg.value(metadata i32 1, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 5, i32 0, i64 -1), !dbg !67
  %3 = add i32 %lsr.iv, -2, !dbg !68
  %call.1 = tail call i32 @bar(i32 noundef %3), !dbg !68
  call void @llvm.dbg.value(metadata i32 %call.1, metadata !53, metadata !DIExpression()), !dbg !70
  %4 = and i32 %call.1, 1, !dbg !71
  %tobool.not.1 = icmp eq i32 %4, 0, !dbg !71
  br i1 %tobool.not.1, label %if.end.1, label %if.then.1, !dbg !73

if.then.1:                                        ; preds = %if.end9
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1), !dbg !74
  %mul4.1 = shl nsw i32 %call.1, 1, !dbg !75
  tail call void @work(i32 noundef %mul4.1), !dbg !76
  br label %if.end.1, !dbg !78

if.end.1:                                         ; preds = %if.then.1, %if.end9
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 7, i32 0, i64 -1), !dbg !79
  %5 = and i32 %call.1, 3, !dbg !81
  %tobool6.not.1 = icmp eq i32 %5, 0, !dbg !81
  br i1 %tobool6.not.1, label %if.end9.1, label %if.then7.1, !dbg !82

if.then7.1:                                       ; preds = %if.end.1
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 8, i32 0, i64 -1), !dbg !83
  %mul8.1 = mul nsw i32 %call.1, 3, !dbg !84
  tail call void @work(i32 noundef %mul8.1), !dbg !85
  br label %if.end9.1, !dbg !87

if.end9.1:                                        ; preds = %if.then7.1, %if.end.1
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 9, i32 0, i64 -1), !dbg !88
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 10, i32 0, i64 -1), !dbg !89
  call void @llvm.dbg.value(metadata i32 2, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !65
  call void @llvm.dbg.value(metadata i32 2, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 5, i32 0, i64 -1), !dbg !67
  %6 = add i32 %lsr.iv, -1, !dbg !68
  %call.2 = tail call i32 @bar(i32 noundef %6), !dbg !68
  call void @llvm.dbg.value(metadata i32 %call.2, metadata !53, metadata !DIExpression()), !dbg !70
  %7 = and i32 %call.2, 1, !dbg !71
  %tobool.not.2 = icmp eq i32 %7, 0, !dbg !71
  br i1 %tobool.not.2, label %if.end.2, label %if.then.2, !dbg !73

if.then.2:                                        ; preds = %if.end9.1
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1), !dbg !74
  %mul4.2 = shl nsw i32 %call.2, 1, !dbg !75
  tail call void @work(i32 noundef %mul4.2), !dbg !76
  br label %if.end.2, !dbg !78

if.end.2:                                         ; preds = %if.then.2, %if.end9.1
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 7, i32 0, i64 -1), !dbg !79
  %8 = and i32 %call.2, 3, !dbg !81
  %tobool6.not.2 = icmp eq i32 %8, 0, !dbg !81
  br i1 %tobool6.not.2, label %if.end9.2, label %if.then7.2, !dbg !82

if.then7.2:                                       ; preds = %if.end.2
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 8, i32 0, i64 -1), !dbg !83
  %mul8.2 = mul nsw i32 %call.2, 3, !dbg !84
  tail call void @work(i32 noundef %mul8.2), !dbg !85
  br label %if.end9.2, !dbg !87

if.end9.2:                                        ; preds = %if.then7.2, %if.end.2
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 9, i32 0, i64 -1), !dbg !88
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 10, i32 0, i64 -1), !dbg !89
  call void @llvm.dbg.value(metadata i32 3, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !65
  call void @llvm.dbg.value(metadata i32 3, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 5, i32 0, i64 -1), !dbg !67
  %call.3 = tail call i32 @bar(i32 noundef %lsr.iv), !dbg !68
  call void @llvm.dbg.value(metadata i32 %call.3, metadata !53, metadata !DIExpression()), !dbg !70
  %9 = and i32 %call.3, 1, !dbg !71
  %tobool.not.3 = icmp eq i32 %9, 0, !dbg !71
  br i1 %tobool.not.3, label %if.end.3, label %if.then.3, !dbg !73

if.then.3:                                        ; preds = %if.end9.2
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 6, i32 0, i64 -1), !dbg !74
  %mul4.3 = shl nsw i32 %call.3, 1, !dbg !75
  tail call void @work(i32 noundef %mul4.3), !dbg !76
  br label %if.end.3, !dbg !78

if.end.3:                                         ; preds = %if.then.3, %if.end9.2
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 7, i32 0, i64 -1), !dbg !79
  %10 = and i32 %call.3, 3, !dbg !81
  %tobool6.not.3 = icmp eq i32 %10, 0, !dbg !81
  br i1 %tobool6.not.3, label %if.end9.3, label %if.then7.3, !dbg !82

if.then7.3:                                       ; preds = %if.end.3
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 8, i32 0, i64 -1), !dbg !83
  %mul8.3 = mul nsw i32 %call.3, 3, !dbg !84
  tail call void @work(i32 noundef %mul8.3), !dbg !85
  br label %if.end9.3, !dbg !87

if.end9.3:                                        ; preds = %if.then7.3, %if.end.3
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 9, i32 0, i64 -1), !dbg !88
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 10, i32 0, i64 -1), !dbg !89
  call void @llvm.dbg.value(metadata i32 4, metadata !51, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !65
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 11, i32 0, i64 -1), !dbg !90
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 12, i32 0, i64 -1), !dbg !92
  call void @llvm.dbg.value(metadata i32 %lsr.iv, metadata !52, metadata !DIExpression(DW_OP_consts, 3, DW_OP_minus, DW_OP_consts, 48, DW_OP_div, DW_OP_consts, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !60
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1), !dbg !61
  %lsr.iv.next = add nuw nsw i32 %lsr.iv, 48, !dbg !93
  %exitcond.not = icmp eq i32 %lsr.iv.next, 2307, !dbg !93
  br i1 %exitcond.not, label %for.end12, label %for.cond1.preheader, !dbg !63, !llvm.loop !95

for.end12:                                        ; preds = %if.end9.3
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 13, i32 0, i64 -1), !dbg !99
  ret void, !dbg !99
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #7

attributes #0 = { nofree noinline nounwind memory(inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nofree noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9}
!llvm.ident = !{!10}
!llvm.pseudo_probe_desc = !{!11, !12, !13, !14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sum", scope: !2, file: !3, line: 7, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git fb16df500443aa5129f4a5e4dc4d9dcac613a809)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "unroll.c", directory: "/home/hoy/build/llvm-github", checksumkind: CSK_MD5, checksum: "11508da575b4d414f8b2f39cf4d90184")
!4 = !{!0}
!5 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"uwtable", i32 2}
!10 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git fb16df500443aa5129f4a5e4dc4d9dcac613a809)"}
!11 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
!12 = !{i64 9204417991963109735, i64 72617220756, !"work"}
!13 = !{i64 6699318081062747564, i64 844700110938769, !"foo"}
!14 = !{i64 -2624081020897602054, i64 281563657672557, !"main"}
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!47 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 15, type: !48, scopeLine: 15, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !50)
!48 = !DISubroutineType(types: !49)
!49 = !{null}
!50 = !{!51, !52, !53}
!51 = !DILocalVariable(name: "i", scope: !47, file: !3, line: 16, type: !18)
!52 = !DILocalVariable(name: "j", scope: !47, file: !3, line: 16, type: !18)
!53 = !DILocalVariable(name: "ii", scope: !54, file: !3, line: 19, type: !18)
!54 = distinct !DILexicalBlock(scope: !55, file: !3, line: 18, column: 30)
!55 = distinct !DILexicalBlock(scope: !56, file: !3, line: 18, column: 6)
!56 = distinct !DILexicalBlock(scope: !57, file: !3, line: 18, column: 6)
!57 = distinct !DILexicalBlock(scope: !58, file: !3, line: 17, column: 4)
!58 = distinct !DILexicalBlock(scope: !47, file: !3, line: 17, column: 4)
!59 = !DILocation(line: 17, column: 11, scope: !58)
!60 = !DILocation(line: 0, scope: !47)
!61 = !DILocation(line: 17, column: 16, scope: !62)
!62 = !DILexicalBlockFile(scope: !57, file: !3, discriminator: 0)
!63 = !DILocation(line: 17, column: 4, scope: !64)
!64 = !DILexicalBlockFile(scope: !58, file: !3, discriminator: 1)
!65 = !DILocation(line: 18, column: 18, scope: !66)
!66 = !DILexicalBlockFile(scope: !55, file: !3, discriminator: 0)
!67 = !DILocation(line: 19, column: 21, scope: !54)
!68 = !DILocation(line: 19, column: 17, scope: !69)
!69 = !DILexicalBlockFile(scope: !54, file: !3, discriminator: 186646647)
!70 = !DILocation(line: 0, scope: !54)
!71 = !DILocation(line: 20, column: 15, scope: !72)
!72 = distinct !DILexicalBlock(scope: !54, file: !3, line: 20, column: 12)
!73 = !DILocation(line: 20, column: 12, scope: !54)
!74 = !DILocation(line: 21, column: 15, scope: !72)
!75 = !DILocation(line: 21, column: 17, scope: !72)
!76 = !DILocation(line: 21, column: 10, scope: !77)
!77 = !DILexicalBlockFile(scope: !72, file: !3, discriminator: 186646655)
!78 = !DILocation(line: 21, column: 10, scope: !72)
!79 = !DILocation(line: 22, column: 12, scope: !80)
!80 = distinct !DILexicalBlock(scope: !54, file: !3, line: 22, column: 12)
!81 = !DILocation(line: 22, column: 15, scope: !80)
!82 = !DILocation(line: 22, column: 12, scope: !54)
!83 = !DILocation(line: 23, column: 15, scope: !80)
!84 = !DILocation(line: 23, column: 17, scope: !80)
!85 = !DILocation(line: 23, column: 10, scope: !86)
!86 = !DILexicalBlockFile(scope: !80, file: !3, discriminator: 186646663)
!87 = !DILocation(line: 23, column: 10, scope: !80)
!88 = !DILocation(line: 24, column: 4, scope: !54)
!89 = !DILocation(line: 18, column: 26, scope: !66)
!90 = !DILocation(line: 24, column: 4, scope: !91)
!91 = !DILexicalBlockFile(scope: !56, file: !3, discriminator: 0)
!92 = !DILocation(line: 17, column: 25, scope: !62)
!93 = !DILocation(line: 17, column: 18, scope: !94)
!94 = !DILexicalBlockFile(scope: !57, file: !3, discriminator: 1)
!95 = distinct !{!95, !96, !97, !98}
!96 = !DILocation(line: 17, column: 4, scope: !58)
!97 = !DILocation(line: 24, column: 4, scope: !58)
!98 = !{!"llvm.loop.mustprogress"}
!99 = !DILocation(line: 25, column: 2, scope: !47)
