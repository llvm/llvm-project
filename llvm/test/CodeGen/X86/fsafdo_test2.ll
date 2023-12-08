; REQUIRES: asserts
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=false < %s | FileCheck %s --check-prefixes=V0,V01
; RUN: llvm-profdata merge --sample -profile-isfs -o %t0.afdo %S/Inputs/fsloader.afdo
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=false -fs-profile-file=%t0.afdo -show-fs-branchprob -disable-ra-fsprofile-loader=false -disable-layout-fsprofile-loader=false < %s 2>&1 | FileCheck %s --check-prefixes=LOADERV0,LOADER
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=true < %s | FileCheck %s --check-prefixes=V1,V01
; RUN: llvm-profdata merge --sample -profile-isfs -o %t1.afdo %S/Inputs/fsloader_v1.afdo
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=true -fs-profile-file=%t1.afdo -show-fs-branchprob -disable-ra-fsprofile-loader=false -disable-layout-fsprofile-loader=false < %s 2>&1 | FileCheck %s --check-prefixes=LOADERV1,LOADER
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=true -fs-profile-file=%S/Inputs/fsloader_v1.afdo -profile-isfs -show-fs-branchprob -disable-ra-fsprofile-loader=false -disable-layout-fsprofile-loader=false < %s 2>&1 | FileCheck %s --check-prefixes=LOADERV1,LOADER
; RUN: llc -enable-fs-discriminator -improved-fs-discriminator=true -fs-profile-file=%S/Inputs/fsloader_v1.afdo -show-fs-branchprob -disable-ra-fsprofile-loader=false -disable-layout-fsprofile-loader=false < %s 2>&1 | FileCheck %s --check-prefixes=NOLOAD
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
;; Check that fs-afdo discriminators are generated.
; V01: .loc    1 23 9 is_stmt 0 discriminator 1 # unroll.c:23:9
; V0: .loc    1 23 9 is_stmt 0 discriminator 3585 # unroll.c:23:9
; V0: .loc    1 23 9 is_stmt 0 discriminator 8705 # unroll.c:23:9
; V0: .loc    1 23 9 is_stmt 0 discriminator 4097 # unroll.c:23:9
; V1: .loc    1 23 9 is_stmt 0 discriminator 257 # unroll.c:23:9
; V1: .loc    1 23 9 is_stmt 0 discriminator 513 # unroll.c:23:9
; V1: .loc    1 23 9 is_stmt 0 discriminator 769 # unroll.c:23:9
;;
;; Check that variable __llvm_fs_discriminator__ is generated.
; V01: .type   __llvm_fs_discriminator__,@object # @__llvm_fs_discriminator__
; V01: .section        .rodata,"a",@progbits
; V01: .weak   __llvm_fs_discriminator__
; V01: __llvm_fs_discriminator__:
; V01: .byte   1
; V01: .size   __llvm_fs_discriminator__, 1

;; Check that new branch probs are generated.
; LOADER: Set branch fs prob: MBB (1 -> 3): unroll.c:22:11-->unroll.c:24:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x7aca7894 / 0x80000000 = 95.93%
; LOADER: Set branch fs prob: MBB (1 -> 2): unroll.c:22:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x0535876c / 0x80000000 = 4.07%
; LOADER: Set branch fs prob: MBB (3 -> 5): unroll.c:24:11-->unroll.c:22:11 W=283590  0x30000000 / 0x80000000 = 37.50% --> 0x7aca7894 / 0x80000000 = 95.93%
; LOADER: Set branch fs prob: MBB (3 -> 4): unroll.c:24:11 W=283590  0x50000000 / 0x80000000 = 62.50% --> 0x0535876c / 0x80000000 = 4.07%
; LOADER: Set branch fs prob: MBB (5 -> 8): unroll.c:22:11-->unroll.c:24:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x021c112e / 0x80000000 = 1.65%
; LOADER: Set branch fs prob: MBB (5 -> 7): unroll.c:22:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x7de3eed2 / 0x80000000 = 98.35%
; LOADER: Set branch fs prob: MBB (8 -> 10): unroll.c:24:11-->unroll.c:22:11 W=283590  0x30000000 / 0x80000000 = 37.50% --> 0x00000000 / 0x80000000 = 0.00%
; LOADER: Set branch fs prob: MBB (8 -> 9): unroll.c:24:11 W=283590  0x50000000 / 0x80000000 = 62.50% --> 0x80000000 / 0x80000000 = 100.00%
; LOADERV0: Set branch fs prob: MBB (10 -> 12): unroll.c:22:11-->unroll.c:24:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x7aca7894 / 0x80000000 = 95.93%
; LOADERV1: Set branch fs prob: MBB (10 -> 12): unroll.c:22:11-->unroll.c:24:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x0a5856e1 / 0x80000000 = 8.08%
; LOADERV0: Set branch fs prob: MBB (10 -> 11): unroll.c:22:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x0535876c / 0x80000000 = 4.07%
; LOADERV1: Set branch fs prob: MBB (10 -> 11): unroll.c:22:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x75a7a91f / 0x80000000 = 91.92%
; LOADER: Set branch fs prob: MBB (12 -> 14): unroll.c:24:11-->unroll.c:22:11 W=283590  0x30000000 / 0x80000000 = 37.50% --> 0x02012507 / 0x80000000 = 1.57%
; LOADER: Set branch fs prob: MBB (12 -> 13): unroll.c:24:11 W=283590  0x50000000 / 0x80000000 = 62.50% --> 0x7dfedaf9 / 0x80000000 = 98.43%
; LOADERV0: Set branch fs prob: MBB (14 -> 16): unroll.c:22:11-->unroll.c:24:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x0a5856e1 / 0x80000000 = 8.08%
; LOADERV1: Set branch fs prob: MBB (14 -> 16): unroll.c:22:11-->unroll.c:24:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x7aca7894 / 0x80000000 = 95.93%
; LOADERV0: Set branch fs prob: MBB (14 -> 15): unroll.c:22:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x75a7a91f / 0x80000000 = 91.92%
; LOADERV1: Set branch fs prob: MBB (14 -> 15): unroll.c:22:11 W=283590  0x40000000 / 0x80000000 = 50.00% --> 0x0535876c / 0x80000000 = 4.07%
; LOADER: Set branch fs prob: MBB (16 -> 18): unroll.c:24:11-->unroll.c:19:3 W=283590  0x30000000 / 0x80000000 = 37.50% --> 0x16588166 / 0x80000000 = 17.46%
; LOADER: Set branch fs prob: MBB (16 -> 17): unroll.c:24:11 W=283590  0x50000000 / 0x80000000 = 62.50% --> 0x69a77e9a / 0x80000000 = 82.54%

;; Check that the profile is not loaded since the reader doesn't know it is a FS profile.
; NOLOAD-NOT: Set branch fs prob

target triple = "x86_64-unknown-linux-gnu"

@sum = dso_local local_unnamed_addr global i32 0, align 4

declare i32 @bar(i32 %i) #0
declare void @work(i32 %i) #2

define dso_local void @foo() #0 !dbg !29 {
entry:
  br label %for.cond1.preheader, !dbg !30

for.cond1.preheader:
  %j.012 = phi i32 [ 0, %entry ], [ %inc11, %if.end9.3 ]
  %mul = mul nuw nsw i32 %j.012, 48
  %call = tail call i32 @bar(i32 %mul), !dbg !32
  %0 = and i32 %call, 1, !dbg !33
  %tobool.not = icmp eq i32 %0, 0, !dbg !33
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !35

if.then:
  %mul4 = shl nsw i32 %call, 1, !dbg !36
  tail call void @work(i32 %mul4), !dbg !37
  br label %if.end, !dbg !38

if.end:
  %1 = and i32 %call, 3, !dbg !39
  %tobool6.not = icmp eq i32 %1, 0, !dbg !39
  br i1 %tobool6.not, label %if.end9, label %if.then7, !dbg !40

if.then7:
  %mul8 = mul nsw i32 %call, 3, !dbg !41
  tail call void @work(i32 %mul8), !dbg !42
  br label %if.end9, !dbg !43

if.end9:
  %add.1 = or i32 %mul, 1, !dbg !44
  %call.1 = tail call i32 @bar(i32 %add.1), !dbg !32
  %2 = and i32 %call.1, 1, !dbg !33
  %tobool.not.1 = icmp eq i32 %2, 0, !dbg !33
  br i1 %tobool.not.1, label %if.end.1, label %if.then.1, !dbg !35

for.end12:
  ret void, !dbg !45

if.then.1:
  %mul4.1 = shl nsw i32 %call.1, 1, !dbg !36
  tail call void @work(i32 %mul4.1), !dbg !37
  br label %if.end.1, !dbg !38

if.end.1:
  %3 = and i32 %call.1, 3, !dbg !39
  %tobool6.not.1 = icmp eq i32 %3, 0, !dbg !39
  br i1 %tobool6.not.1, label %if.end9.1, label %if.then7.1, !dbg !40

if.then7.1:
  %mul8.1 = mul nsw i32 %call.1, 3, !dbg !41
  tail call void @work(i32 %mul8.1), !dbg !42
  br label %if.end9.1, !dbg !43

if.end9.1:
  %add.2 = or i32 %mul, 2, !dbg !44
  %call.2 = tail call i32 @bar(i32 %add.2), !dbg !32
  %4 = and i32 %call.2, 1, !dbg !33
  %tobool.not.2 = icmp eq i32 %4, 0, !dbg !33
  br i1 %tobool.not.2, label %if.end.2, label %if.then.2, !dbg !35

if.then.2:
  %mul4.2 = shl nsw i32 %call.2, 1, !dbg !36
  tail call void @work(i32 %mul4.2), !dbg !37
  br label %if.end.2, !dbg !38

if.end.2:
  %5 = and i32 %call.2, 3, !dbg !39
  %tobool6.not.2 = icmp eq i32 %5, 0, !dbg !39
  br i1 %tobool6.not.2, label %if.end9.2, label %if.then7.2, !dbg !40

if.then7.2:
  %mul8.2 = mul nsw i32 %call.2, 3, !dbg !41
  tail call void @work(i32 %mul8.2), !dbg !42
  br label %if.end9.2, !dbg !43

if.end9.2:
  %add.3 = or i32 %mul, 3, !dbg !44
  %call.3 = tail call i32 @bar(i32 %add.3), !dbg !32
  %6 = and i32 %call.3, 1, !dbg !33
  %tobool.not.3 = icmp eq i32 %6, 0, !dbg !33
  br i1 %tobool.not.3, label %if.end.3, label %if.then.3, !dbg !35

if.then.3:
  %mul4.3 = shl nsw i32 %call.3, 1, !dbg !36
  tail call void @work(i32 %mul4.3), !dbg !37
  br label %if.end.3, !dbg !38

if.end.3:
  %7 = and i32 %call.3, 3, !dbg !39
  %tobool6.not.3 = icmp eq i32 %7, 0, !dbg !39
  br i1 %tobool6.not.3, label %if.end9.3, label %if.then7.3, !dbg !40

if.then7.3:
  %mul8.3 = mul nsw i32 %call.3, 3, !dbg !41
  tail call void @work(i32 %mul8.3), !dbg !42
  br label %if.end9.3, !dbg !43

if.end9.3:
  %inc11 = add nuw nsw i32 %j.012, 1, !dbg !46
  %exitcond.not = icmp eq i32 %inc11, 48, !dbg !48
  br i1 %exitcond.not, label %for.end12, label %for.cond1.preheader, !dbg !30, !llvm.loop !49
}


attributes #0 = { noinline nounwind uwtable "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nofree noinline norecurse nounwind uwtable "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "unroll.c", directory: "a/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, column: 3, scope: !7)
!10 = !DILocation(line: 5, column: 5, scope: !7)
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 6, column: 10, scope: !7)
!16 = !DILocation(line: 7, column: 1, scope: !7)
!17 = !DILocation(line: 6, column: 3, scope: !18)
!18 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 1)
!19 = distinct !DISubprogram(name: "work", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 11, column: 7, scope: !19)
!21 = !DILocation(line: 11, column: 11, scope: !22)
!22 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 1)
!23 = !DILocation(line: 11, column: 11, scope: !24)
!24 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 2)
!25 = !DILocation(line: 11, column: 7, scope: !26)
!26 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 3)
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocation(line: 15, column: 1, scope: !19)
!29 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 17, type: !8, scopeLine: 17, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!30 = !DILocation(line: 19, column: 3, scope: !31)
!31 = !DILexicalBlockFile(scope: !29, file: !1, discriminator: 2)
!32 = !DILocation(line: 21, column: 16, scope: !31)
!33 = !DILocation(line: 22, column: 14, scope: !34)
!34 = !DILexicalBlockFile(scope: !29, file: !1, discriminator: 1)
!35 = !DILocation(line: 22, column: 11, scope: !31)
!36 = !DILocation(line: 23, column: 16, scope: !29)
!37 = !DILocation(line: 23, column: 9, scope: !34)
!38 = !DILocation(line: 23, column: 9, scope: !31)
!39 = !DILocation(line: 24, column: 14, scope: !34)
!40 = !DILocation(line: 24, column: 11, scope: !31)
!41 = !DILocation(line: 25, column: 16, scope: !29)
!42 = !DILocation(line: 25, column: 9, scope: !34)
!43 = !DILocation(line: 25, column: 9, scope: !31)
!44 = !DILocation(line: 21, column: 21, scope: !34)
!45 = !DILocation(line: 27, column: 1, scope: !29)
!46 = !DILocation(line: 19, column: 24, scope: !47)
!47 = !DILexicalBlockFile(scope: !29, file: !1, discriminator: 3)
!48 = !DILocation(line: 19, column: 17, scope: !34)
!49 = distinct !{!49, !50, !51}
!50 = !DILocation(line: 19, column: 3, scope: !29)
!51 = !DILocation(line: 26, column: 3, scope: !29)
!52 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 29, type: !8, scopeLine: 29, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!53 = !DILocation(line: 31, column: 3, scope: !54)
!54 = !DILexicalBlockFile(scope: !52, file: !1, discriminator: 2)
!55 = !DILocation(line: 32, column: 5, scope: !52)
!56 = !DILocation(line: 31, column: 30, scope: !57)
!57 = !DILexicalBlockFile(scope: !52, file: !1, discriminator: 3)
!58 = !DILocation(line: 31, column: 17, scope: !59)
!59 = !DILexicalBlockFile(scope: !52, file: !1, discriminator: 1)
!60 = distinct !{!60, !61, !62}
!61 = !DILocation(line: 31, column: 3, scope: !52)
!62 = !DILocation(line: 33, column: 3, scope: !52)
!63 = !DILocation(line: 34, column: 1, scope: !52)
