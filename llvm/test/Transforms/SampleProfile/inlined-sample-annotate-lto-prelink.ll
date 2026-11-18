; ModuleID = 'src/caller.cpp'

; // clang++ -O2 -g -fpseudo-probe-for-profiling -flto=thin -Xclang -disable-llvm-passes -S -emit-llvm -DV2 -fprofile-sample-use=prof.txt ${SRC} -o ${OUT}.fe.ll

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
; int wrapper(int n);
;
; // caller.cpp
; #include "lib.h"
; int caller(int n) {
;   int s = 0;
;   for (int i = 0; i < n; ++i) s += wrapper(i & 63);
;   return s;
; }
;
; // heavy.cpp
; #include "lib.h"
; #define S4 r = (r ^ n) * 33; r = (r ^ n) * 33; r = (r ^ n) * 33; r = (r ^ n) * 33;
; template <class T> int heavy(int n) { int r = n; S4 S4 S4 S4 S4 S4 S4 S4 S4 S4 return r; }
; template int heavy<lib::Tag>(int);
; int wrapper(int n) { return heavy<lib::Tag>(n); }

; REQUIRES: x86-registered-target
; RUN: opt -passes='pseudo-probe,thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/inlined-sample-annotate-lto-prelink.prof --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %s -S -o %t1.prelink.ll
; RUN: opt -passes='pseudo-probe,thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/inlined-sample-annotate-lto-prelink.prof --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %S/Inputs/inlined-sample-annotate-lto-prelink-callees.ll -S -o %t2.prelink.ll 2>&1 | FileCheck %s --check-prefix=PRELINK
; RUN: opt -module-summary %t1.prelink.ll -o %t1.o
; RUN: opt -module-summary %t2.prelink.ll -o %t2.o
; RUN: llvm-lto2 run -thinlto-distributed-indexes -o %t/lto2 %t1.o %t2.o -r %t1.o,_Z6calleri,px -r %t1.o,_Z7wrapperi, -r %t2.o,_Z5heavyIN3lib2v23TagEEii,px -r %t2.o,_Z7wrapperi,px
; RUN: opt -passes=function-import,sample-profile -summary-file=%t1.o.thinlto.bc -sample-profile-file=%S/Inputs/inlined-sample-annotate-lto-prelink.prof -pass-remarks=sample-profile --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl %t1.o -S -o %t1.opt.ll 2>&1 | FileCheck %s --check-prefix=POSTLINK

; PRELINK: Function _Z5heavyIN3lib2v23TagEEii is not in profile or profile symbol list.
; PRELINK: Run stale profile matching for _Z7wrapperi
; PRELINK: The functions _Z5heavyIN3lib2v23TagEEii(IR) and _Z5heavyIN3lib3TagEEii(Profile) share the same base name: heavy.
; PRELINK: Function:_Z5heavyIN3lib2v23TagEEii matches profile:_Z5heavyIN3lib3TagEEii
; PRELINK: Processing Function _Z7wrapperi
; PRELINK: %call = call noundef i32 @_Z5heavyIN3lib2v23TagEEii(i32 noundef %n), !dbg !263 - weight: 1000000 - factor: 1.00)

; POSTLINK: remark: src/caller.cpp:4:36: '_Z7wrapperi' inlined into '_Z6calleri' to match profiling context with (cost=always): preinliner at callsite _Z6calleri:2:36;
; POSTLINK: remark: src/heavy.cpp:11:29: '_Z5heavyIN3lib2v23TagEEii' inlined into '_Z6calleri' to match profiling context with (cost=always): preinliner at callsite _Z7wrapperi:0:29 @ _Z6calleri:2:36;

source_filename = "src/caller.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z6calleri(i32 noundef %n) #0 !dbg !11 {
entry:
  %n.addr = alloca i32, align 4
  %s = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4, !tbaa !20
    #dbg_declare(ptr %n.addr, !16, !DIExpression(), !21)
  call void @llvm.lifetime.start.p0(ptr %s) #3, !dbg !22
    #dbg_declare(ptr %s, !17, !DIExpression(), !23)
  store i32 0, ptr %s, align 4, !dbg !24, !tbaa !20
  call void @llvm.lifetime.start.p0(ptr %i) #3, !dbg !25
    #dbg_declare(ptr %i, !18, !DIExpression(), !26)
  store i32 0, ptr %i, align 4, !dbg !27, !tbaa !20
  br label %for.cond, !dbg !25

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !28, !tbaa !20
  %1 = load i32, ptr %n.addr, align 4, !dbg !30, !tbaa !20
  %cmp = icmp slt i32 %0, %1, !dbg !31
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !32

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0(ptr %i) #3, !dbg !33
  br label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %i, align 4, !dbg !34, !tbaa !20
  %and = and i32 %2, 63, !dbg !35
  %call = call noundef i32 @_Z7wrapperi(i32 noundef %and), !dbg !36
  %3 = load i32, ptr %s, align 4, !dbg !37, !tbaa !20
  %add = add nsw i32 %3, %call, !dbg !38
  store i32 %add, ptr %s, align 4, !dbg !39, !tbaa !20
  br label %for.inc, !dbg !40

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4, !dbg !41, !tbaa !20
  %inc = add nsw i32 %4, 1, !dbg !42
  store i32 %inc, ptr %i, align 4, !dbg !43, !tbaa !20
  br label %for.cond, !dbg !33, !llvm.loop !44

for.end:                                          ; preds = %for.cond.cleanup
  %5 = load i32, ptr %s, align 4, !dbg !48, !tbaa !20
  call void @llvm.lifetime.end.p0(ptr %s) #3, !dbg !49
  ret i32 %5, !dbg !50
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare !dbg !51 dso_local noundef i32 @_Z7wrapperi(i32 noundef) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}
!llvm.errno.tbaa = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (git@github.com:HighW4y2H3ll/llvm-project.git 671765346dbba0840bb06b6141d5456460619754)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "src/caller.cpp", directory: "/home/h2h/test/repro_import_guid_tiny", checksumkind: CSK_MD5, checksum: "631e25beceb54b435aee909a76532d42")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 24.0.0git (git@github.com:HighW4y2H3ll/llvm-project.git 671765346dbba0840bb06b6141d5456460619754)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"__libc_errno", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !DISubprogram(name: "caller", linkageName: "_Z6calleri", scope: !1, file: !1, line: 2, type: !12, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15, keyInstructions: true)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16, !17, !18}
!16 = !DILocalVariable(name: "n", arg: 1, scope: !11, file: !1, line: 2, type: !14)
!17 = !DILocalVariable(name: "s", scope: !11, file: !1, line: 3, type: !14)
!18 = !DILocalVariable(name: "i", scope: !19, file: !1, line: 4, type: !14)
!19 = distinct !DILexicalBlock(scope: !11, file: !1, line: 4, column: 3)
!20 = !{!8, !8, i64 0}
!21 = !DILocation(line: 2, column: 16, scope: !11)
!22 = !DILocation(line: 3, column: 3, scope: !11)
!23 = !DILocation(line: 3, column: 7, scope: !11)
!24 = !DILocation(line: 3, column: 7, scope: !11, atomGroup: 1, atomRank: 1)
!25 = !DILocation(line: 4, column: 8, scope: !19)
!26 = !DILocation(line: 4, column: 12, scope: !19)
!27 = !DILocation(line: 4, column: 12, scope: !19, atomGroup: 2, atomRank: 1)
!28 = !DILocation(line: 4, column: 19, scope: !29)
!29 = distinct !DILexicalBlock(scope: !19, file: !1, line: 4, column: 3)
!30 = !DILocation(line: 4, column: 23, scope: !29)
!31 = !DILocation(line: 4, column: 21, scope: !29, atomGroup: 3, atomRank: 1)
!32 = !DILocation(line: 4, column: 3, scope: !19, atomGroup: 4, atomRank: 1)
!33 = !DILocation(line: 4, column: 3, scope: !29)
!34 = !DILocation(line: 4, column: 44, scope: !29)
!35 = !DILocation(line: 4, column: 46, scope: !29)
!36 = !DILocation(line: 4, column: 36, scope: !29)
!37 = !DILocation(line: 4, column: 33, scope: !29)
!38 = !DILocation(line: 4, column: 33, scope: !29, atomGroup: 5, atomRank: 2)
!39 = !DILocation(line: 4, column: 33, scope: !29, atomGroup: 5, atomRank: 1)
!40 = !DILocation(line: 4, column: 31, scope: !29, atomGroup: 7, atomRank: 1)
!41 = !DILocation(line: 4, column: 26, scope: !29)
!42 = !DILocation(line: 4, column: 26, scope: !29, atomGroup: 6, atomRank: 2)
!43 = !DILocation(line: 4, column: 26, scope: !29, atomGroup: 6, atomRank: 1)
!44 = distinct !{!44, !45, !46, !47}
!45 = !DILocation(line: 4, column: 3, scope: !19)
!46 = !DILocation(line: 4, column: 50, scope: !19)
!47 = !{!"llvm.loop.mustprogress"}
!48 = !DILocation(line: 5, column: 10, scope: !11, atomGroup: 9, atomRank: 2)
!49 = !DILocation(line: 6, column: 1, scope: !11)
!50 = !DILocation(line: 5, column: 3, scope: !11, atomGroup: 9, atomRank: 1)
!51 = !DISubprogram(name: "wrapper", linkageName: "_Z7wrapperi", scope: !52, file: !52, line: 14, type: !12, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!52 = !DIFile(filename: "src/lib.h", directory: "/home/h2h/test/repro_import_guid_tiny", checksumkind: CSK_MD5, checksum: "c696fbf8e2efefb4f774df1237cd7af1")
