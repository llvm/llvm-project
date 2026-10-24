; *** IR Dump Before SampleProfileLoaderPass on [module] ***
; ModuleID = 'test.cc'

; // clang++ test.cc -fpseudo-probe-for-profiling -fdebug-info-for-profiling -mllvm -print-before=sample-profile -fprofile-sample-use=/dev/null
; 
; __attribute__((always_inline)) void F(int i) {
;   i = i%2;
; }
; 
; void A() {
;   F(1);
; }
; 
; void B() {
;   F(2);
; }

; REQUIRES: x86-registered-target
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/inlined-sample-annotate.prof -o %t
; RUN: opt < %s -S -salvage-unused-profile -sample-profile-file=%t -persist-profile-staleness -passes="sample-profile" | FileCheck %s
; CHECK: define dso_local void @_Z1Fi(i32 noundef %i) #0 !dbg !39 !prof !42 {
; CHECK: define dso_local void @_Z1Av() #1 !dbg !47 !prof !42 {
; CHECK: !38 = !{!"NumStaleProfileFunc", i64 1, !"TotalProfiledFunc", i64 1, !"MismatchedFunctionSamples", i64 22222, !"TotalFunctionSamples", i64 22222, !"NumCallGraphRecoveredProfiledFunc", i64 1, !"NumCallGraphRecoveredFuncSamples", i64 22222, !"NumMismatchedCallsites", i64 0, !"NumRecoveredCallsites", i64 0, !"TotalProfiledCallsites", i64 1, !"MismatchedCallsiteSamples", i64 0, !"RecoveredCallsiteSamples", i64 0}
; CHECK: !42 = !{!"function_entry_count", i64 11111}


source_filename = "test.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

; Function Attrs: alwaysinline mustprogress nounwind uwtable
define dso_local void @_Z1Fi(i32 noundef %i) #0 !dbg !4 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  call void @llvm.pseudoprobe(i64 307639317667852319, i64 1, i32 0, i64 -1), !dbg !9
  %0 = load i32, ptr %i.addr, align 4, !dbg !9
  %rem = srem i32 %0, 2, !dbg !8
  store i32 %rem, ptr %i.addr, align 4, !dbg !7
  ret void, !dbg !11
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z1Av() #1 !dbg !12 {
entry:
  call void @llvm.pseudoprobe(i64 8005135552184634117, i64 1, i32 0, i64 -1), !dbg !14
  call void @_Z1Fi(i32 noundef 1), !dbg !28
  ret void, !dbg !16
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z1Bv() #1 !dbg !17 {
entry:
  call void @llvm.pseudoprobe(i64 -7776078424693146369, i64 1, i32 0, i64 -1), !dbg !19
  call void @_Z1Fi(i32 noundef 2), !dbg !31
  ret void, !dbg !21
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #2

attributes #0 = { alwaysinline mustprogress nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!22, !23, !24}
!llvm.ident = !{!25}
!llvm.pseudo_probe_desc = !{!26, !29, !32}

!0 = !DIFile(filename: "test.cc", directory: "/home/h2h/llvm-build_2")
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !0, producer: "clang version 24.0.0git (https://github.com/llvm/llvm-project.git ceabc54809682ba79a490fa6f9f81cdb30523279)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!2 = !{}
!3 = !DISubroutineType(types: !2)
!4 = distinct !DISubprogram(name: "F", linkageName: "_Z1Fi", scope: !0, file: !0, line: 3, type: !3, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1)
!7 = !DILocation(line: 4, column: 5, scope: !4)
!8 = !DILocation(line: 4, column: 8, scope: !4)
!9 = !DILocation(line: 4, column: 7, scope: !4)
!11 = !DILocation(line: 5, column: 1, scope: !4)
!12 = distinct !DISubprogram(name: "A", linkageName: "_Z1Av", scope: !0, file: !0, line: 7, type: !3, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1)
!14 = !DILocation(line: 8, column: 5, scope: !12)
!16 = !DILocation(line: 9, column: 1, scope: !12)
!17 = distinct !DISubprogram(name: "B", linkageName: "_Z1Bv", scope: !0, file: !0, line: 11, type: !3, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1)
!19 = !DILocation(line: 12, column: 5, scope: !17)
!21 = !DILocation(line: 13, column: 1, scope: !17)
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 7, !"uwtable", i32 2}
!24 = !{i32 7, !"frame-pointer", i32 2}
!25 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git ceabc54809682ba79a490fa6f9f81cdb30523279)"}
!26 = !{i64 307639317667852319, i64 4294967295, !"_Z1Fi"}
!27 = !DILexicalBlockFile(scope: !12, file: !0, discriminator: 455082007)
!28 = !DILocation(line: 8, column: 5, scope: !27)
!29 = !{i64 8005135552184634117, i64 281479271677951, !"_Z1Av"}
!30 = !DILexicalBlockFile(scope: !17, file: !0, discriminator: 455082007)
!31 = !DILocation(line: 12, column: 5, scope: !30)
!32 = !{i64 -7776078424693146369, i64 281479271677951, !"_Z1Bv"}
