; ModuleID = 'src/caller.cpp'

; // clang++ -O2 -fdebug-info-for-profiling -fpseudo-probe-for-profiling -flto=thin -Xclang -disable-llvm-passes -S -emit-llvm -DV2 -fprofile-sample-use=prof.txt ${SRC} -o ${OUT}.fe.ll

; // lib.h
; #pragma once
; namespace lib {
; #ifdef V2
; inline namespace v2 {
; #endif
; struct Tag {};
; #ifdef V2
; }
; #endif
; }
; template <class T> int heavy(int n);
;
; // caller.cpp
; #include "lib.h"
; int caller(int n) {
;   int s = 0;
;   for (int i = 0; i < n; ++i) s += heavy<lib::Tag>(i & 63);
;   return s;
; }
;
; // heavy.cpp
; #include "lib.h"
; template <class T> int heavy(int n) { return (n ^ 5) * 33; }
; template int heavy<lib::Tag>(int);

; REQUIRES: x86-registered-target
; REQUIRES: asserts
; RUN: opt -passes='pseudo-probe,thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/inlined-sample-annotate-lto-prelink-nowrap.prof --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %s -S -o %t1.prelink.ll 2>&1 | FileCheck %s --check-prefix=PRELINK
; RUN: cat %t1.prelink.ll | FileCheck %s --check-prefix=PRELINK-GUID
; RUN: opt -passes='pseudo-probe,thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/inlined-sample-annotate-lto-prelink-nowrap.prof --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %S/Inputs/inlined-sample-annotate-lto-prelink-nowrap-callees.ll -S -o %t2.prelink.ll
; RUN: opt -module-summary %t1.prelink.ll -o %t1.o
; RUN: opt -module-summary %t2.prelink.ll -o %t2.o
; RUN: llvm-lto2 run -thinlto-distributed-indexes -import-hot-multiplier=0 -o %t/lto2 %t1.o %t2.o -r %t1.o,_Z6calleri,px -r %t1.o,_Z5heavyIN3lib2v23TagEEii, -r %t2.o,_Z5heavyIN3lib2v23TagEEii,px
; RUN: opt -passes=function-import,sample-profile -summary-file=%t1.o.thinlto.bc -sample-profile-file=%S/Inputs/inlined-sample-annotate-lto-prelink-nowrap.prof -pass-remarks=sample-profile --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %t1.o -S -o %t1.opt.ll 2>&1 | FileCheck %s --check-prefix=POSTLINK

; PRELINK: Function _Z5heavyIN3lib2v23TagEEii is not in profile or profile symbol list.
; PRELINK: The functions _Z5heavyIN3lib2v23TagEEii(IR) and _Z5heavyIN3lib3TagEEii(Profile) share the same base name: heavy.
; PRELINK: Function:_Z5heavyIN3lib2v23TagEEii matches profile:_Z5heavyIN3lib3TagEEii

; PRELINK-GUID: define dso_local noundef i32 @_Z6calleri(i32 noundef %n) local_unnamed_addr #0 !dbg !40 !prof !43 !guid !44 {
; PRELINK-GUID: !43 = !{!"function_entry_count", i64 1000000, i64 -7245025397704601654}

; POSTLINK: remark: src/caller.cpp:4:36: '_Z5heavyIN3lib2v23TagEEii' inlined into '_Z6calleri' to match profiling context with (cost=always): preinliner at callsite _Z6calleri:2:36;

; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/inlined-sample-annotate-lto-prelink-nowrap-cs.prof -o %t.extbin
; RUN: opt -passes='pseudo-probe,thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.extbin --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %s -S -o %t1.prelink.ll 2>&1 | FileCheck %s --check-prefix=PRELINK-CS
; RUN: cat %t1.prelink.ll | FileCheck %s --check-prefix=PRELINK-GUID-CS
; RUN: opt -passes='pseudo-probe,thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%t.extbin --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %S/Inputs/inlined-sample-annotate-lto-prelink-nowrap-callees.ll -S -o %t2.prelink.ll
; RUN: opt -module-summary %t1.prelink.ll -o %t1.o
; RUN: opt -module-summary %t2.prelink.ll -o %t2.o
; RUN: llvm-lto2 run -thinlto-distributed-indexes -import-hot-multiplier=0 -o %t/lto2 %t1.o %t2.o -r %t1.o,_Z6calleri,px -r %t1.o,_Z5heavyIN3lib2v23TagEEii, -r %t2.o,_Z5heavyIN3lib2v23TagEEii,px
; RUN: opt -passes=function-import,sample-profile -summary-file=%t1.o.thinlto.bc -sample-profile-file=%t.extbin -pass-remarks=sample-profile --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %t1.o -S -o %t1.opt.ll 2>&1 | FileCheck %s --check-prefix=POSTLINK-CS

; PRELINK-CS: Function _Z5heavyIN3lib2v23TagEEii is not in profile or profile symbol list.
; PRELINK-CS: Direct basename match: _Z5heavyIN3lib2v23TagEEii (IR) -> _Z5heavyIN3lib3TagEEii (Profile) [basename: heavy]
; PRELINK-CS: Direct basename matching found 1 matches

; PRELINK-GUID-CS: define dso_local noundef i32 @_Z6calleri(i32 noundef %n) local_unnamed_addr #0 !dbg !40 !prof !43 !guid !44 {
; PRELINK-GUID-CS: !43 = !{!"function_entry_count", i64 1000000, i64 -7245025397704601654}

; POSTLINK-CS: remark: src/caller.cpp:4:36: '_Z5heavyIN3lib2v23TagEEii' inlined into '_Z6calleri' to match profiling context with (cost=-25, threshold=3000) at callsite _Z6calleri:2:36;

source_filename = "src/caller.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z6calleri(i32 noundef %n) #0 !dbg !10 {
entry:
  %n.addr = alloca i32, align 4
  %s = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr %s) #3, !dbg !14
  store i32 0, ptr %s, align 4, !dbg !15, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr %i) #3, !dbg !16
  store i32 0, ptr %i, align 4, !dbg !17, !tbaa !13
  br label %for.cond, !dbg !16

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !18, !tbaa !13
  %1 = load i32, ptr %n.addr, align 4, !dbg !19, !tbaa !13
  %cmp = icmp slt i32 %0, %1, !dbg !20
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !21

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0(ptr %i) #3, !dbg !21
  br label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %i, align 4, !dbg !22, !tbaa !13
  %and = and i32 %2, 63, !dbg !23
  %call = call noundef i32 @_Z5heavyIN3lib2v23TagEEii(i32 noundef %and), !dbg !24
  %3 = load i32, ptr %s, align 4, !dbg !25, !tbaa !13
  %add = add nsw i32 %3, %call, !dbg !25
  store i32 %add, ptr %s, align 4, !dbg !25, !tbaa !13
  br label %for.inc, !dbg !26

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4, !dbg !27, !tbaa !13
  %inc = add nsw i32 %4, 1, !dbg !27
  store i32 %inc, ptr %i, align 4, !dbg !27, !tbaa !13
  br label %for.cond, !dbg !21, !llvm.loop !28

for.end:                                          ; preds = %for.cond.cleanup
  %5 = load i32, ptr %s, align 4, !dbg !31, !tbaa !13
  call void @llvm.lifetime.end.p0(ptr %s) #3, !dbg !32
  ret i32 %5, !dbg !33
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare dso_local noundef i32 @_Z5heavyIN3lib2v23TagEEii(i32 noundef) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (https://github.com/llvm/llvm-project.git 6bcf1c81a057579159e38dc03adf0afdd51c9c29)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "src/caller.cpp", directory: "/home/h2h/test/repro_import_guid_nowrapper")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git 6bcf1c81a057579159e38dc03adf0afdd51c9c29)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"__libc_errno", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !DISubprogram(name: "caller", linkageName: "_Z6calleri", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !{!7, !7, i64 0}
!14 = !DILocation(line: 3, column: 3, scope: !10)
!15 = !DILocation(line: 3, column: 7, scope: !10)
!16 = !DILocation(line: 4, column: 8, scope: !10)
!17 = !DILocation(line: 4, column: 12, scope: !10)
!18 = !DILocation(line: 4, column: 19, scope: !10)
!19 = !DILocation(line: 4, column: 23, scope: !10)
!20 = !DILocation(line: 4, column: 21, scope: !10)
!21 = !DILocation(line: 4, column: 3, scope: !10)
!22 = !DILocation(line: 4, column: 52, scope: !10)
!23 = !DILocation(line: 4, column: 54, scope: !10)
!24 = !DILocation(line: 4, column: 36, scope: !10)
!25 = !DILocation(line: 4, column: 33, scope: !10)
!26 = !DILocation(line: 4, column: 31, scope: !10)
!27 = !DILocation(line: 4, column: 26, scope: !10)
!28 = distinct !{!28, !21, !29, !30}
!29 = !DILocation(line: 4, column: 58, scope: !10)
!30 = !{!"llvm.loop.mustprogress"}
!31 = !DILocation(line: 5, column: 10, scope: !10)
!32 = !DILocation(line: 6, column: 1, scope: !10)
!33 = !DILocation(line: 5, column: 3, scope: !10)
