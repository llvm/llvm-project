; RUN: llvm-profdata merge --sample -profile-isfs -o %t.afdo %S/Inputs/fsloader.afdo
; RUN: llc -enable-fs-discriminator -fs-profile-file=%t.afdo -disable-ra-fsprofile-loader=false -disable-layout-fsprofile-loader=false -print-machine-bfi -print-bfi-func-name=foo -print-before=fs-profile-loader -stop-after=fs-profile-loader < %s 2>&1 | FileCheck %s --check-prefix=BFI
;
;;
;; C source code for the test (compiler at -O3):
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
;; Check BFI before and after

; BFI: block-frequency-info: foo
; BFI:  - BB0[entry]: float = 1.0, int = 8, count = 4268
; BFI:  - BB1[for.cond1.preheader]: float = 59.967, int = 479, count = 255547
; BFI:  - BB2[if.then]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB3[if.end]: float = 59.967, int = 479, count = 255547
; BFI:  - BB4[if.then7]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB5[if.end9]: float = 59.967, int = 479, count = 255547
; BFI:  - BB6[if.then.1]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB7[if.end.1]: float = 59.967, int = 479, count = 255547
; BFI:  - BB8[if.then7.1]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB9[if.end9.1]: float = 59.967, int = 479, count = 255547
; BFI:  - BB10[if.then.2]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB11[if.end.2]: float = 59.967, int = 479, count = 255547
; BFI:  - BB12[if.then7.2]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB13[if.end9.2]: float = 59.967, int = 479, count = 255547
; BFI:  - BB14[if.then.3]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB15[if.end.3]: float = 59.967, int = 479, count = 255547
; BFI:  - BB16[if.then7.3]: float = 2.5405, int = 20, count = 10670
; BFI:  - BB17[if.end9.3]: float = 59.967, int = 479, count = 255547
; BFI:  - BB18[for.end12]: float = 1.0, int = 8, count = 4268
;
; BFI: # *** IR Dump Before SampleFDO loader in MIR (fs-profile-loader) ***:
; BFI: # End machine code for function foo.
;
; BFI: block-frequency-info: foo
; BFI:  - BB0[entry]: float = 1.0, int = 8, count = 4268
; BFI:  - BB1[for.cond1.preheader]: float = 66.446, int = 531, count = 283289
; BFI:  - BB2[if.then]: float = 2.7041, int = 21, count = 11204
; BFI:  - BB3[if.end]: float = 66.446, int = 531, count = 283289
; BFI:  - BB4[if.then7]: float = 2.7041, int = 21, count = 11204
; BFI:  - BB5[if.end9]: float = 66.446, int = 531, count = 283289
; BFI:  - BB6[if.then.1]: float = 65.351, int = 522, count = 278487
; BFI:  - BB7[if.end.1]: float = 66.446, int = 531, count = 283289
; BFI:  - BB8[if.then7.1]: float = 66.446, int = 531, count = 283289
; BFI:  - BB9[if.end9.1]: float = 66.446, int = 531, count = 283289
; BFI:  - BB10[if.then.2]: float = 2.7041, int = 21, count = 11204
; BFI:  - BB11[if.end.2]: float = 66.446, int = 531, count = 283289
; BFI:  - BB12[if.then7.2]: float = 65.405, int = 523, count = 279021
; BFI:  - BB13[if.end9.2]: float = 66.446, int = 531, count = 283289
; BFI:  - BB14[if.then.3]: float = 61.075, int = 488, count = 260348
; BFI:  - BB15[if.end.3]: float = 66.446, int = 531, count = 283289
; BFI:  - BB16[if.then7.3]: float = 54.846, int = 438, count = 233673
; BFI:  - BB17[if.end9.3]: float = 66.446, int = 531, count = 283289
; BFI:  - BB18[for.end12]: float = 1.0, int = 8, count = 4268

target triple = "x86_64-unknown-linux-gnu"

@sum = dso_local local_unnamed_addr global i32 0, align 4

declare i32 @bar(i32 %i) #0
declare void @work(i32 %i) #2
declare i32 @main() #3

define dso_local void @foo() local_unnamed_addr #0 !dbg !61 !prof !62 {
entry:
  br label %for.cond1.preheader, !dbg !63

for.cond1.preheader:
  %j.024 = phi i32 [ 0, %entry ], [ %inc11, %if.end9.3 ]
  %mul = mul nuw nsw i32 %j.024, 48
  %call = tail call i32 @bar(i32 %mul), !dbg !65, !prof !66
  %0 = and i32 %call, 1, !dbg !67
  %tobool.not = icmp eq i32 %0, 0, !dbg !67
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !68, !prof !69

if.then:
  %mul4 = shl nsw i32 %call, 1, !dbg !70
  tail call void @work(i32 %mul4), !dbg !71, !prof !72
  br label %if.end, !dbg !71

if.end:
  %1 = and i32 %call, 3, !dbg !73
  %tobool6.not = icmp eq i32 %1, 0, !dbg !73
  br i1 %tobool6.not, label %if.end9, label %if.then7, !dbg !74, !prof !69

if.then7:
  %mul8 = mul nsw i32 %call, 3, !dbg !75
  tail call void @work(i32 %mul8), !dbg !76, !prof !72
  br label %if.end9, !dbg !76

if.end9:
  %add.1 = or i32 %mul, 1, !dbg !77
  %call.1 = tail call i32 @bar(i32 %add.1), !dbg !65, !prof !66
  %2 = and i32 %call.1, 1, !dbg !67
  %tobool.not.1 = icmp eq i32 %2, 0, !dbg !67
  br i1 %tobool.not.1, label %if.end.1, label %if.then.1, !dbg !68, !prof !69

if.then.1:
  %mul4.1 = shl nsw i32 %call.1, 1, !dbg !70
  tail call void @work(i32 %mul4.1), !dbg !71, !prof !72
  br label %if.end.1, !dbg !71

if.end.1:
  %3 = and i32 %call.1, 3, !dbg !73
  %tobool6.not.1 = icmp eq i32 %3, 0, !dbg !73
  br i1 %tobool6.not.1, label %if.end9.1, label %if.then7.1, !dbg !74, !prof !69

if.then7.1:
  %mul8.1 = mul nsw i32 %call.1, 3, !dbg !75
  tail call void @work(i32 %mul8.1), !dbg !76, !prof !72
  br label %if.end9.1, !dbg !76

if.end9.1:
  %add.2 = or i32 %mul, 2, !dbg !77
  %call.2 = tail call i32 @bar(i32 %add.2), !dbg !65, !prof !66
  %4 = and i32 %call.2, 1, !dbg !67
  %tobool.not.2 = icmp eq i32 %4, 0, !dbg !67
  br i1 %tobool.not.2, label %if.end.2, label %if.then.2, !dbg !68, !prof !69

if.then.2:
  %mul4.2 = shl nsw i32 %call.2, 1, !dbg !70
  tail call void @work(i32 %mul4.2), !dbg !71, !prof !72
  br label %if.end.2, !dbg !71

if.end.2:
  %5 = and i32 %call.2, 3, !dbg !73
  %tobool6.not.2 = icmp eq i32 %5, 0, !dbg !73
  br i1 %tobool6.not.2, label %if.end9.2, label %if.then7.2, !dbg !74, !prof !69

if.then7.2:
  %mul8.2 = mul nsw i32 %call.2, 3, !dbg !75
  tail call void @work(i32 %mul8.2), !dbg !76, !prof !72
  br label %if.end9.2, !dbg !76

if.end9.2:
  %add.3 = or i32 %mul, 3, !dbg !77
  %call.3 = tail call i32 @bar(i32 %add.3), !dbg !65, !prof !66
  %6 = and i32 %call.3, 1, !dbg !67
  %tobool.not.3 = icmp eq i32 %6, 0, !dbg !67
  br i1 %tobool.not.3, label %if.end.3, label %if.then.3, !dbg !68, !prof !69

if.then.3:
  %mul4.3 = shl nsw i32 %call.3, 1, !dbg !70
  tail call void @work(i32 %mul4.3), !dbg !71, !prof !72
  br label %if.end.3, !dbg !71

if.end.3:
  %7 = and i32 %call.3, 3, !dbg !73
  %tobool6.not.3 = icmp eq i32 %7, 0, !dbg !73
  br i1 %tobool6.not.3, label %if.end9.3, label %if.then7.3, !dbg !74, !prof !69

if.then7.3:
  %mul8.3 = mul nsw i32 %call.3, 3, !dbg !75
  tail call void @work(i32 %mul8.3), !dbg !76, !prof !72
  br label %if.end9.3, !dbg !76

if.end9.3:
  %inc11 = add nuw nsw i32 %j.024, 1, !dbg !78
  %exitcond.not = icmp eq i32 %inc11, 48, !dbg !80
  br i1 %exitcond.not, label %for.end12, label %for.cond1.preheader, !dbg !63, !prof !81, !llvm.loop !82

for.end12:
  ret void, !dbg !86
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind uwtable willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !35}
!llvm.ident = !{!40}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 755f5e23159796d727c3d95d60894a52eb675b1b)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/aaa.c", directory: "/mnt/ssd/xur/llvm_dev/gitwork/llvm-project/rel")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 1, !"ProfileSummary", !7}
!7 = !{!8, !9, !10, !11, !12, !13, !14, !15, !16, !17}
!8 = !{!"ProfileFormat", !"SampleProfile"}
!9 = !{!"TotalCount", i64 9484871}
!10 = !{!"MaxCount", i64 1246499}
!11 = !{!"MaxInternalCount", i64 0}
!12 = !{!"MaxFunctionCount", i64 1346190}
!13 = !{!"NumCounts", i64 31}
!14 = !{!"NumFunctions", i64 4}
!15 = !{!"IsPartialProfile", i64 0}
!16 = !{!"PartialProfileRatio", double 0.000000e+00}
!17 = !{!"DetailedSummary", !18}
!18 = !{!19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34}
!19 = !{i32 10000, i64 1246499, i32 2}
!20 = !{i32 100000, i64 1246499, i32 2}
!21 = !{i32 200000, i64 1246499, i32 2}
!22 = !{i32 300000, i64 1056020, i32 4}
!23 = !{i32 400000, i64 1056020, i32 4}
!24 = !{i32 500000, i64 283590, i32 6}
!25 = !{i32 600000, i64 279149, i32 9}
!26 = !{i32 700000, i64 278916, i32 12}
!27 = !{i32 800000, i64 269485, i32 15}
!28 = !{i32 900000, i64 260670, i32 19}
!29 = !{i32 950000, i64 234082, i32 22}
!30 = !{i32 990000, i64 234082, i32 22}
!31 = !{i32 999000, i64 4156, i32 27}
!32 = !{i32 999900, i64 4045, i32 29}
!33 = !{i32 999990, i64 4045, i32 29}
!34 = !{i32 999999, i64 4045, i32 29}
!35 = !{i32 5, !"CG Profile", !36}
!36 = !{!37, !38, !39}
!37 = !{ptr @foo, ptr @bar, i64 1022188}
!38 = !{ptr @foo, ptr @work, i64 85360}
!39 = !{ptr @main, ptr @foo, i64 2080}
!40 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 755f5e23159796d727c3d95d60894a52eb675b1b)"}
!41 = distinct !DISubprogram(name: "bar", scope: !42, file: !42, line: 3, type: !43, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !44)
!42 = !DIFile(filename: "/tmp/aaa.c", directory: "")
!43 = !DISubroutineType(types: !44)
!44 = !{}
!45 = !{!"function_entry_count", i64 1076806}
!46 = !DILocation(line: 4, column: 3, scope: !41)
!47 = !DILocation(line: 5, column: 5, scope: !41)
!48 = !{!49, !49, i64 0}
!49 = !{!"int", !50, i64 0}
!50 = !{!"omnipotent char", !51, i64 0}
!51 = !{!"Simple C/C++ TBAA"}
!52 = !DILocation(line: 6, column: 10, scope: !41)
!53 = !DILocation(line: 7, column: 1, scope: !41)
!54 = !DILocation(line: 6, column: 3, scope: !41)
!55 = distinct !DISubprogram(name: "work", scope: !42, file: !42, line: 10, type: !43, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !44)
!56 = !{!"function_entry_count", i64 1346191}
!57 = !DILocation(line: 11, column: 7, scope: !55)
!58 = !DILocation(line: 11, column: 11, scope: !55)
!59 = !DILocation(line: 0, scope: !55)
!60 = !DILocation(line: 15, column: 1, scope: !55)
!61 = distinct !DISubprogram(name: "foo", scope: !42, file: !42, line: 17, type: !43, scopeLine: 17, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !44)
!62 = !{!"function_entry_count", i64 4268}
!63 = !DILocation(line: 19, column: 3, scope: !64)
!64 = !DILexicalBlockFile(scope: !61, file: !42, discriminator: 1)
!65 = !DILocation(line: 21, column: 16, scope: !61)
!66 = !{!"branch_weights", i32 272442}
!67 = !DILocation(line: 22, column: 14, scope: !61)
!68 = !DILocation(line: 22, column: 11, scope: !61)
!69 = !{!"branch_weights", i32 260902, i32 11542}
!70 = !DILocation(line: 23, column: 16, scope: !61)
!71 = !DILocation(line: 23, column: 9, scope: !61)
!72 = !{!"branch_weights", i32 11541}
!73 = !DILocation(line: 24, column: 14, scope: !61)
!74 = !DILocation(line: 24, column: 11, scope: !61)
!75 = !DILocation(line: 25, column: 16, scope: !61)
!76 = !DILocation(line: 25, column: 9, scope: !61)
!77 = !DILocation(line: 21, column: 21, scope: !61)
!78 = !DILocation(line: 19, column: 24, scope: !79)
!79 = !DILexicalBlockFile(scope: !61, file: !42, discriminator: 2)
!80 = !DILocation(line: 19, column: 17, scope: !64)
!81 = !{!"branch_weights", i32 4269, i32 251732}
!82 = distinct !{!82, !83, !84, !85}
!83 = !DILocation(line: 19, column: 3, scope: !61)
!84 = !DILocation(line: 26, column: 3, scope: !61)
!85 = !{!"llvm.loop.mustprogress"}
!86 = !DILocation(line: 27, column: 1, scope: !61)
!87 = distinct !DISubprogram(name: "main", scope: !42, file: !42, line: 29, type: !43, scopeLine: 29, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !44)
!88 = !{!"function_entry_count", i64 1}
!89 = !DILocation(line: 31, column: 3, scope: !90)
!90 = !DILexicalBlockFile(scope: !87, file: !42, discriminator: 1)
!91 = !DILocation(line: 32, column: 5, scope: !87)
!92 = !{!"branch_weights", i32 4156}
!93 = !DILocation(line: 31, column: 30, scope: !94)
!94 = !DILexicalBlockFile(scope: !87, file: !42, discriminator: 2)
!95 = !DILocation(line: 31, column: 17, scope: !90)
!96 = !{!"branch_weights", i32 2, i32 4157}
!97 = distinct !{!97, !98, !99, !85}
!98 = !DILocation(line: 31, column: 3, scope: !87)
!99 = !DILocation(line: 33, column: 3, scope: !87)
!100 = !DILocation(line: 34, column: 1, scope: !87)
