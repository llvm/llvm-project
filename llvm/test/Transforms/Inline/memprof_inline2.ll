;; Test for memprof metadata propagation, ensuring metadata is moved to
;; inlined callsites.
;; Also check that callsite metadata was updated with inlined stack ids.
;;
;; The following code was used to generate the following IR and its memprof
;; profile:
;;
;; #include <stdlib.h>
;; #include <string.h>
;; #include <unistd.h>
;; char *foo() {
;;   return new char[10];
;; }
;; char *foo2() __attribute((noinline)) {
;;   return foo();
;; }
;; char *bar() {
;;   return foo2();
;; }
;; char *baz() {
;;   return foo2();
;; }
;; int main(int argc, char **argv) {
;;   char *c = foo();
;;   char *d = foo();
;;   char *e = bar();
;;   char *f = baz();
;;   memset(c, 0, 10);
;;   memset(d, 0, 10);
;;   memset(e, 0, 10);
;;   memset(f, 0, 10);
;;   delete[] c;
;;   sleep(200);
;;   delete[] d;
;;   delete[] e;
;;   delete[] f;
;;   return 0;
;; }

; RUN: opt -inline %s -S | FileCheck %s

; ModuleID = 'memprof_inline2.cc'
source_filename = "memprof_inline2.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress uwtable
; CHECK-LABEL: define dso_local noundef i8* @_Z3foov
define dso_local noundef i8* @_Z3foov() #0 !dbg !39 {
entry:
  ;; We should still have memprof/callsite metadata for the non-inlined calls
  ;; from main, but should have removed those from the inlined call in_Z4foo2v.
  ;; CHECK: call {{.*}} @_Znam{{.*}} !memprof ![[ORIGMEMPROF:[0-9]+]]
  %call = call noalias noundef nonnull i8* @_Znam(i64 noundef 10) #7, !dbg !42, !memprof !43, !callsite !52
  ret i8* %call, !dbg !53
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znam(i64 noundef) #1

;; Mark noinline so we don't inline into calls from bar and baz. We should end
;; up with a memprof metadata on the call to foo below.
; Function Attrs: mustprogress noinline uwtable
; CHECK-LABEL: define dso_local noundef i8* @_Z4foo2v
define dso_local noundef i8* @_Z4foo2v() #2 !dbg !54 {
entry:
  ;; We should have memprof metadata for the call stacks from bar and baz,
  ;; and the callsite metadata should be the concatentation of the id from the
  ;; inlined call to new and the original callsite.
  ; CHECK: call {{.*}} @_Znam{{.*}} !memprof ![[NEWMEMPROF:[0-9]+]], !callsite ![[NEWCALLSITE:[0-9]+]]
  %call = call noundef i8* @_Z3foov(), !dbg !55, !callsite !56
  ret i8* %call, !dbg !57
}

; Function Attrs: mustprogress uwtable
define dso_local noundef i8* @_Z3barv() #0 !dbg !58 {
entry:
  %call = call noundef i8* @_Z4foo2v(), !dbg !59, !callsite !60
  ret i8* %call, !dbg !61
}

; Function Attrs: mustprogress uwtable
define dso_local noundef i8* @_Z3bazv() #0 !dbg !62 {
entry:
  %call = call noundef i8* @_Z4foo2v(), !dbg !63, !callsite !64
  ret i8* %call, !dbg !65
}

;; Make sure we don't propagate any memprof/callsite metadata
; Function Attrs: mustprogress uwtable
; CHECK-LABEL: define dso_local noundef i8* @notprofiled
define dso_local noundef i8* @notprofiled() #0 !dbg !66 {
entry:
  ; CHECK: call {{.*}} @_Znam
  ; CHECK-NOT: !memprof
  ; CHECK-NOT: !callsite
  %call = call noundef i8* @_Z3foov(), !dbg !67
  ; CHECK-NEXT: ret
  ret i8* %call, !dbg !68
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main(i32 noundef %argc, i8** noundef %argv) #3 !dbg !69 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %c = alloca i8*, align 8
  %d = alloca i8*, align 8
  %e = alloca i8*, align 8
  %f = alloca i8*, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  ;; The below 4 callsites are all annotated as noinline
  %call = call noundef i8* @_Z3foov() #8, !dbg !70, !callsite !71
  store i8* %call, i8** %c, align 8, !dbg !72
  %call1 = call noundef i8* @_Z3foov() #8, !dbg !73, !callsite !74
  store i8* %call1, i8** %d, align 8, !dbg !75
  %call2 = call noundef i8* @_Z3barv() #8, !dbg !76, !callsite !77
  store i8* %call2, i8** %e, align 8, !dbg !78
  %call3 = call noundef i8* @_Z3bazv() #8, !dbg !79, !callsite !80
  store i8* %call3, i8** %f, align 8, !dbg !81
  %0 = load i8*, i8** %c, align 8, !dbg !82
  call void @llvm.memset.p0i8.i64(i8* align 1 %0, i8 0, i64 10, i1 false), !dbg !83
  %1 = load i8*, i8** %d, align 8, !dbg !84
  call void @llvm.memset.p0i8.i64(i8* align 1 %1, i8 0, i64 10, i1 false), !dbg !85
  %2 = load i8*, i8** %e, align 8, !dbg !86
  call void @llvm.memset.p0i8.i64(i8* align 1 %2, i8 0, i64 10, i1 false), !dbg !87
  %3 = load i8*, i8** %f, align 8, !dbg !88
  call void @llvm.memset.p0i8.i64(i8* align 1 %3, i8 0, i64 10, i1 false), !dbg !89
  %4 = load i8*, i8** %c, align 8, !dbg !90
  %isnull = icmp eq i8* %4, null, !dbg !91
  br i1 %isnull, label %delete.end, label %delete.notnull, !dbg !91

delete.notnull:                                   ; preds = %entry
  call void @_ZdaPv(i8* noundef %4) #9, !dbg !92
  br label %delete.end, !dbg !92

delete.end:                                       ; preds = %delete.notnull, %entry
  %call4 = call i32 @sleep(i32 noundef 200), !dbg !94
  %5 = load i8*, i8** %d, align 8, !dbg !95
  %isnull5 = icmp eq i8* %5, null, !dbg !96
  br i1 %isnull5, label %delete.end7, label %delete.notnull6, !dbg !96

delete.notnull6:                                  ; preds = %delete.end
  call void @_ZdaPv(i8* noundef %5) #9, !dbg !97
  br label %delete.end7, !dbg !97

delete.end7:                                      ; preds = %delete.notnull6, %delete.end
  %6 = load i8*, i8** %e, align 8, !dbg !98
  %isnull8 = icmp eq i8* %6, null, !dbg !99
  br i1 %isnull8, label %delete.end10, label %delete.notnull9, !dbg !99

delete.notnull9:                                  ; preds = %delete.end7
  call void @_ZdaPv(i8* noundef %6) #9, !dbg !100
  br label %delete.end10, !dbg !100

delete.end10:                                     ; preds = %delete.notnull9, %delete.end7
  %7 = load i8*, i8** %f, align 8, !dbg !101
  %isnull11 = icmp eq i8* %7, null, !dbg !102
  br i1 %isnull11, label %delete.end13, label %delete.notnull12, !dbg !102

delete.notnull12:                                 ; preds = %delete.end10
  call void @_ZdaPv(i8* noundef %7) #9, !dbg !103
  br label %delete.end13, !dbg !103

delete.end13:                                     ; preds = %delete.notnull12, %delete.end10
  ret i32 0, !dbg !104
}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(i8* noundef) #5

declare i32 @sleep(i32 noundef) #6

attributes #0 = { mustprogress uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress noinline norecurse optnone uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { argmemonly nofree nounwind willreturn writeonly }
attributes #5 = { nobuiltin nounwind "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { builtin allocsize(0) }
attributes #8 = { noinline }
attributes #9 = { builtin nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8, !9}
!llvm.ident = !{!38}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git e09c924f98ec157adeaa74819b0aec9a07a1b552)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof_inline.cc", directory: "/usr/local/google/home/tejohnson/llvm/tmp", checksumkind: CSK_MD5, checksum: "8711f6fd269e6cb5611fef48bc906eab")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{i32 1, !"ProfileSummary", !10}
!10 = !{!11, !12, !13, !14, !15, !16, !17, !18, !19, !20}
!11 = !{!"ProfileFormat", !"InstrProf"}
!12 = !{!"TotalCount", i64 0}
!13 = !{!"MaxCount", i64 0}
!14 = !{!"MaxInternalCount", i64 0}
!15 = !{!"MaxFunctionCount", i64 0}
!16 = !{!"NumCounts", i64 0}
!17 = !{!"NumFunctions", i64 0}
!18 = !{!"IsPartialProfile", i64 0}
!19 = !{!"PartialProfileRatio", double 0.000000e+00}
!20 = !{!"DetailedSummary", !21}
!21 = !{!22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37}
!22 = !{i32 10000, i64 0, i32 0}
!23 = !{i32 100000, i64 0, i32 0}
!24 = !{i32 200000, i64 0, i32 0}
!25 = !{i32 300000, i64 0, i32 0}
!26 = !{i32 400000, i64 0, i32 0}
!27 = !{i32 500000, i64 0, i32 0}
!28 = !{i32 600000, i64 0, i32 0}
!29 = !{i32 700000, i64 0, i32 0}
!30 = !{i32 800000, i64 0, i32 0}
!31 = !{i32 900000, i64 0, i32 0}
!32 = !{i32 950000, i64 0, i32 0}
!33 = !{i32 990000, i64 0, i32 0}
!34 = !{i32 999000, i64 0, i32 0}
!35 = !{i32 999900, i64 0, i32 0}
!36 = !{i32 999990, i64 0, i32 0}
!37 = !{i32 999999, i64 0, i32 0}
!38 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git e09c924f98ec157adeaa74819b0aec9a07a1b552)"}
!39 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !40, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!40 = !DISubroutineType(types: !41)
!41 = !{}
!42 = !DILocation(line: 5, column: 10, scope: !39)
;; The first 2 are from the direct calls to foo from main. Those stay on the
;; callsite in foo, which isn't inlined into main due to the callsites in main
;; being annotated as noinline.
;; The second 2 are from the calls from foo2, which inlines its callsite to foo
;; but is not itself inlined into its callers. Therefore they get moved to a
;; new memprof metadata within foo2.
!43 = !{!44, !46, !48, !50}
!44 = !{!45, !"cold"}
!45 = !{i64 -2458008693472584243, i64 7394638144382192936}
!46 = !{!47, !"noncold"}
!47 = !{i64 -2458008693472584243, i64 -8908997186479157179}
!48 = !{!49, !"noncold"}
!49 = !{i64 -2458008693472584243, i64 -8079659623765193173, i64 -4805294506621015872}
!50 = !{!51, !"cold"}
!51 = !{i64 -2458008693472584243, i64 -8079659623765193173, i64 -972865200055133905}
; CHECK: ![[ORIGMEMPROF]] = !{![[ORIGMIB1:[0-9]+]], ![[ORIGMIB2:[0-9]+]]}
; CHECK: ![[ORIGMIB1]] = !{![[ORIGMIBSTACK1:[0-9]+]], !"cold"}
; CHECK: ![[ORIGMIBSTACK1]] = !{i64 -2458008693472584243, i64 7394638144382192936}
; CHECK: ![[ORIGMIB2]] = !{![[ORIGMIBSTACK2:[0-9]+]], !"notcold"}
; CHECK: ![[ORIGMIBSTACK2]] = !{i64 -2458008693472584243, i64 -8908997186479157179}
; CHECK: ![[NEWMEMPROF]] = !{![[NEWMIB1:[0-9]+]], ![[NEWMIB2:[0-9]+]]}
; CHECK: ![[NEWMIB1]] = !{![[NEWMIBSTACK1:[0-9]+]], !"notcold"}
; CHECK: ![[NEWMIBSTACK1]] = !{i64 -2458008693472584243, i64 -8079659623765193173, i64 -4805294506621015872}
; CHECK: ![[NEWMIB2]] = !{![[NEWMIBSTACK2:[0-9]+]], !"cold"}
; CHECK: ![[NEWMIBSTACK2]] = !{i64 -2458008693472584243, i64 -8079659623765193173, i64 -972865200055133905}
; CHECK: ![[NEWCALLSITE]] = !{i64 -2458008693472584243, i64 -8079659623765193173}
!52 = !{i64 -2458008693472584243}
!53 = !DILocation(line: 5, column: 3, scope: !39)
!54 = distinct !DISubprogram(name: "foo2", linkageName: "_Z4foo2v", scope: !1, file: !1, line: 7, type: !40, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!55 = !DILocation(line: 8, column: 10, scope: !54)
!56 = !{i64 -8079659623765193173}
!57 = !DILocation(line: 8, column: 3, scope: !54)
!58 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 10, type: !40, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!59 = !DILocation(line: 11, column: 10, scope: !58)
!60 = !{i64 -972865200055133905}
!61 = !DILocation(line: 11, column: 3, scope: !58)
!62 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 13, type: !40, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!63 = !DILocation(line: 14, column: 10, scope: !62)
!64 = !{i64 -4805294506621015872}
!65 = !DILocation(line: 14, column: 3, scope: !62)
!66 = distinct !DISubprogram(name: "notprofiled", linkageName: "notprofiled", scope: !1, file: !1, line: 400, type: !40, scopeLine: 400, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!67 = !DILocation(line: 401, column: 10, scope: !66)
!68 = !DILocation(line: 401, column: 3, scope: !66)
!69 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 16, type: !40, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!70 = !DILocation(line: 17, column: 13, scope: !69)
!71 = !{i64 -8908997186479157179}
!72 = !DILocation(line: 17, column: 9, scope: !69)
!73 = !DILocation(line: 18, column: 13, scope: !69)
!74 = !{i64 7394638144382192936}
!75 = !DILocation(line: 18, column: 9, scope: !69)
!76 = !DILocation(line: 19, column: 13, scope: !69)
!77 = !{i64 -5510257407004945023}
!78 = !DILocation(line: 19, column: 9, scope: !69)
!79 = !DILocation(line: 20, column: 13, scope: !69)
!80 = !{i64 8771588133652501463}
!81 = !DILocation(line: 20, column: 9, scope: !69)
!82 = !DILocation(line: 21, column: 10, scope: !69)
!83 = !DILocation(line: 21, column: 3, scope: !69)
!84 = !DILocation(line: 22, column: 10, scope: !69)
!85 = !DILocation(line: 22, column: 3, scope: !69)
!86 = !DILocation(line: 23, column: 10, scope: !69)
!87 = !DILocation(line: 23, column: 3, scope: !69)
!88 = !DILocation(line: 24, column: 10, scope: !69)
!89 = !DILocation(line: 24, column: 3, scope: !69)
!90 = !DILocation(line: 25, column: 12, scope: !69)
!91 = !DILocation(line: 25, column: 3, scope: !69)
!92 = !DILocation(line: 25, column: 3, scope: !93)
!93 = !DILexicalBlockFile(scope: !69, file: !1, discriminator: 2)
!94 = !DILocation(line: 26, column: 3, scope: !69)
!95 = !DILocation(line: 27, column: 12, scope: !69)
!96 = !DILocation(line: 27, column: 3, scope: !69)
!97 = !DILocation(line: 27, column: 3, scope: !93)
!98 = !DILocation(line: 28, column: 12, scope: !69)
!99 = !DILocation(line: 28, column: 3, scope: !69)
!100 = !DILocation(line: 28, column: 3, scope: !93)
!101 = !DILocation(line: 29, column: 12, scope: !69)
!102 = !DILocation(line: 29, column: 3, scope: !69)
!103 = !DILocation(line: 29, column: 3, scope: !93)
!104 = !DILocation(line: 30, column: 3, scope: !69)
