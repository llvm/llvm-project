;; Test for memprof metadata propagation, ensuring metadata is simplified
;; to function attributes appropriately after inlining profiled call chains.
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
;; char *foo2() {
;;   return foo();
;; }
;; int main(int argc, char **argv) {
;;   char *c = foo();
;;   char *d = foo();
;;   char *e = foo2();
;;   memset(c, 0, 10);
;;   memset(d, 0, 10);
;;   memset(e, 0, 10);
;;   delete[] c;
;;   sleep(200);
;;   delete[] d;
;;   delete[] e;
;;   return 0;
;; }


; RUN: opt -inline %s -S | FileCheck %s

; ModuleID = 'memprof_inline.cc'
source_filename = "memprof_inline.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress uwtable
; CHECK-LABEL: define dso_local noundef i8* @_Z3foov
define dso_local noundef i8* @_Z3foov() #0 !dbg !39 {
entry:
  ; CHECK: call {{.*}} @_Znam
  ; CHECK-NOT: !memprof
  ; CHECK-NOT: !callsite
  %call = call noalias noundef nonnull i8* @_Znam(i64 noundef 10) #6, !dbg !42, !memprof !43, !callsite !50
  ; CHECK-NEXT: ret
  ret i8* %call, !dbg !51
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znam(i64 noundef) #1

; Function Attrs: mustprogress uwtable
; CHECK-LABEL: define dso_local noundef i8* @_Z4foo2v
define dso_local noundef i8* @_Z4foo2v() #0 !dbg !52 {
entry:
  ; CHECK: call {{.*}} @_Znam{{.*}} #[[COLD:[0-9]+]]
  %call = call noundef i8* @_Z3foov(), !dbg !53, !callsite !54
  ret i8* %call, !dbg !55
}

; Function Attrs: mustprogress norecurse uwtable
; CHECK-LABEL: define dso_local noundef i32 @main
define dso_local noundef i32 @main(i32 noundef %argc, i8** noundef %argv) #2 !dbg !56 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %c = alloca i8*, align 8
  %d = alloca i8*, align 8
  %e = alloca i8*, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  ; CHECK: call {{.*}} @_Znam{{.*}} #[[NOTCOLD:[0-9]+]]
  %call = call noundef i8* @_Z3foov(), !dbg !57, !callsite !58
  store i8* %call, i8** %c, align 8, !dbg !59
  ; CHECK: call {{.*}} @_Znam{{.*}} #[[COLD]]
  %call1 = call noundef i8* @_Z3foov(), !dbg !60, !callsite !61
  store i8* %call1, i8** %d, align 8, !dbg !62
  ; CHECK: call {{.*}} @_Znam{{.*}} #[[COLD]]
  %call2 = call noundef i8* @_Z4foo2v(), !dbg !63, !callsite !64
  store i8* %call2, i8** %e, align 8, !dbg !65
  %0 = load i8*, i8** %c, align 8, !dbg !66
  call void @llvm.memset.p0i8.i64(i8* align 1 %0, i8 0, i64 10, i1 false), !dbg !67
  %1 = load i8*, i8** %d, align 8, !dbg !68
  call void @llvm.memset.p0i8.i64(i8* align 1 %1, i8 0, i64 10, i1 false), !dbg !69
  %2 = load i8*, i8** %e, align 8, !dbg !70
  call void @llvm.memset.p0i8.i64(i8* align 1 %2, i8 0, i64 10, i1 false), !dbg !71
  %3 = load i8*, i8** %c, align 8, !dbg !72
  %isnull = icmp eq i8* %3, null, !dbg !73
  br i1 %isnull, label %delete.end, label %delete.notnull, !dbg !73

delete.notnull:                                   ; preds = %entry
  call void @_ZdaPv(i8* noundef %3) #7, !dbg !74
  br label %delete.end, !dbg !74

delete.end:                                       ; preds = %delete.notnull, %entry
  %call4 = call i32 @sleep(i32 noundef 200), !dbg !76
  %4 = load i8*, i8** %d, align 8, !dbg !77
  %isnull5 = icmp eq i8* %4, null, !dbg !78
  br i1 %isnull5, label %delete.end7, label %delete.notnull6, !dbg !78

delete.notnull6:                                  ; preds = %delete.end
  call void @_ZdaPv(i8* noundef %4) #7, !dbg !79
  br label %delete.end7, !dbg !79

delete.end7:                                      ; preds = %delete.notnull6, %delete.end
  %5 = load i8*, i8** %e, align 8, !dbg !80
  %isnull8 = icmp eq i8* %5, null, !dbg !81
  br i1 %isnull8, label %delete.end10, label %delete.notnull9, !dbg !81

delete.notnull9:                                  ; preds = %delete.end7
  call void @_ZdaPv(i8* noundef %5) #7, !dbg !82
  br label %delete.end10, !dbg !82

delete.end10:                                     ; preds = %delete.notnull9, %delete.end7
  ret i32 0, !dbg !83
}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(i8* noundef) #4

declare i32 @sleep(i32 noundef) #5

; CHECK: attributes #[[COLD]] = { builtin allocsize(0) "memprof"="cold" }
; CHECK: attributes #[[NOTCOLD]] = { builtin allocsize(0) "memprof"="notcold" }

attributes #0 = { mustprogress uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress norecurse uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { argmemonly nofree nounwind willreturn writeonly }
attributes #4 = { nobuiltin nounwind "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { builtin allocsize(0) }
attributes #7 = { builtin nounwind }

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
!43 = !{!44, !46, !48}
!44 = !{!45, !"cold"}
!45 = !{i64 -2458008693472584243, i64 7394638144382192936}
!46 = !{!47, !"noncold"}
!47 = !{i64 -2458008693472584243, i64 -8908997186479157179}
!48 = !{!49, !"cold"}
!49 = !{i64 -2458008693472584243, i64 -8079659623765193173}
!50 = !{i64 -2458008693472584243}
!51 = !DILocation(line: 5, column: 3, scope: !39)
!52 = distinct !DISubprogram(name: "foo2", linkageName: "_Z4foo2v", scope: !1, file: !1, line: 7, type: !40, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!53 = !DILocation(line: 8, column: 10, scope: !52)
!54 = !{i64 -8079659623765193173}
!55 = !DILocation(line: 8, column: 3, scope: !52)
!56 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 16, type: !40, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
!57 = !DILocation(line: 17, column: 13, scope: !56)
!58 = !{i64 -8908997186479157179}
!59 = !DILocation(line: 17, column: 9, scope: !56)
!60 = !DILocation(line: 18, column: 13, scope: !56)
!61 = !{i64 7394638144382192936}
!62 = !DILocation(line: 18, column: 9, scope: !56)
!63 = !DILocation(line: 19, column: 13, scope: !56)
!64 = !{i64 -5510257407004945023}
!65 = !DILocation(line: 19, column: 9, scope: !56)
!66 = !DILocation(line: 21, column: 10, scope: !56)
!67 = !DILocation(line: 21, column: 3, scope: !56)
!68 = !DILocation(line: 22, column: 10, scope: !56)
!69 = !DILocation(line: 22, column: 3, scope: !56)
!70 = !DILocation(line: 23, column: 10, scope: !56)
!71 = !DILocation(line: 23, column: 3, scope: !56)
!72 = !DILocation(line: 25, column: 12, scope: !56)
!73 = !DILocation(line: 25, column: 3, scope: !56)
!74 = !DILocation(line: 25, column: 3, scope: !75)
!75 = !DILexicalBlockFile(scope: !56, file: !1, discriminator: 2)
!76 = !DILocation(line: 26, column: 3, scope: !56)
!77 = !DILocation(line: 27, column: 12, scope: !56)
!78 = !DILocation(line: 27, column: 3, scope: !56)
!79 = !DILocation(line: 27, column: 3, scope: !75)
!80 = !DILocation(line: 28, column: 12, scope: !56)
!81 = !DILocation(line: 28, column: 3, scope: !56)
!82 = !DILocation(line: 28, column: 3, scope: !75)
!83 = !DILocation(line: 30, column: 3, scope: !56)
