; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/non-probe-stale-profile-matching-template.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -profile-isfs 2>&1 | FileCheck %s

; Test that stale profile matching correctly distinguishes template
; specializations by preserving template arguments in the demangled base name.
; The IR has ns2::foo<int> and ns2::foo<double>, while the profile was collected
; from ns1::foo<int> and ns1::foo<double>. The matcher should pair each IR
; function with the profile entry that has the same template arguments, not
; conflate them under the bare name "foo".

; Verify orphan functions are detected.
; CHECK-DAG: Function _ZN3ns23fooIiEET_S1_ is not in profile or profile symbol list.
; CHECK-DAG: Function _ZN3ns23fooIdEET_S1_ is not in profile or profile symbol list.

; Verify basename matching preserves template arguments: foo<int> matches
; foo<int>, foo<double> matches foo<double>, and they are not conflated.
; CHECK-DAG: The functions _ZN3ns23fooIiEET_S1_(IR) and _ZN3ns13fooIiEET_S1_(Profile) share the same base name: foo<int>.
; CHECK-DAG: Function:_ZN3ns23fooIiEET_S1_ matches profile:_ZN3ns13fooIiEET_S1_
; CHECK-DAG: The functions _ZN3ns23fooIdEET_S1_(IR) and _ZN3ns13fooIdEET_S1_(Profile) share the same base name: foo<double>.
; CHECK-DAG: Function:_ZN3ns23fooIdEET_S1_ matches profile:_ZN3ns13fooIdEET_S1_

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 1, align 4

define dso_local i32 @_Z3bari(i32 noundef %p) #0 !dbg !9 {
entry:
  ret i32 %p, !dbg !13
}

define dso_local i32 @_ZN3ns23fooIiEET_S1_(i32 noundef %x) #0 !dbg !14 {
entry:
  %add = add nsw i32 %x, 1, !dbg !15
  ret i32 %add, !dbg !16
}

define dso_local double @_ZN3ns23fooIdEET_S1_(double noundef %x) #0 !dbg !17 {
entry:
  %add = fadd double %x, 1.000000e+00, !dbg !20
  ret double %add, !dbg !21
}

define dso_local i32 @main() #1 !dbg !22 {
entry:
  br label %for.cond, !dbg !23

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !24
  %cmp = icmp slt i32 %i.0, 1000000, !dbg !25
  br i1 %cmp, label %for.body, label %for.end, !dbg !26

for.body:
  %0 = load volatile i32, ptr @x, align 4, !dbg !27
  %call = call i32 @_ZN3ns23fooIiEET_S1_(i32 noundef %0), !dbg !28
  %1 = load volatile i32, ptr @x, align 4, !dbg !29
  %add = add nsw i32 %1, %call, !dbg !29
  store volatile i32 %add, ptr @x, align 4, !dbg !29
  %2 = load volatile i32, ptr @x, align 4, !dbg !30
  %call2 = call i32 @_Z3bari(i32 noundef %2), !dbg !31
  %3 = load volatile i32, ptr @x, align 4, !dbg !32
  %add3 = add nsw i32 %3, %call2, !dbg !32
  store volatile i32 %add3, ptr @x, align 4, !dbg !32
  %4 = load volatile i32, ptr @x, align 4, !dbg !33
  %conv = sitofp i32 %4 to double, !dbg !33
  %call4 = call double @_ZN3ns23fooIdEET_S1_(double noundef %conv), !dbg !34
  %conv5 = fptosi double %call4 to i32, !dbg !34
  %5 = load volatile i32, ptr @x, align 4, !dbg !35
  %add6 = add nsw i32 %5, %conv5, !dbg !35
  store volatile i32 %add6, ptr @x, align 4, !dbg !35
  %6 = load volatile i32, ptr @x, align 4, !dbg !36
  %call7 = call i32 @_Z3bari(i32 noundef %6), !dbg !37
  %7 = load volatile i32, ptr @x, align 4, !dbg !38
  %add8 = add nsw i32 %7, %call7, !dbg !38
  store volatile i32 %add8, ptr @x, align 4, !dbg !38
  %inc = add nsw i32 %i.0, 1, !dbg !39
  br label %for.cond, !dbg !40, !llvm.loop !41

for.end:
  ret i32 0, !dbg !44
}

attributes #0 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 19.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "path")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 19.0.0git"}
!9 = distinct !DISubprogram(name: "bar", linkageName: "_Z3bari", scope: !10, file: !10, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DIFile(filename: "test.cpp", directory: "path")
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 3, column: 3, scope: !9)
!14 = distinct !DISubprogram(name: "foo<int>", linkageName: "_ZN3ns23fooIiEET_S1_", scope: !10, file: !10, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 7, column: 12, scope: !14)
!16 = !DILocation(line: 7, column: 3, scope: !14)
!17 = distinct !DISubprogram(name: "foo<double>", linkageName: "_ZN3ns23fooIdEET_S1_", scope: !10, file: !10, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = !DILocation(line: 7, column: 12, scope: !17)
!21 = !DILocation(line: 7, column: 3, scope: !17)
!22 = distinct !DISubprogram(name: "main", scope: !10, file: !10, line: 11, type: !11, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = !DILocation(line: 12, column: 3, scope: !22)
!24 = !DILocation(line: 12, scope: !22)
!25 = !DILocation(line: 12, column: 21, scope: !22)
!26 = !DILocation(line: 12, column: 3, scope: !22)
!27 = !DILocation(line: 13, column: 18, scope: !22)
!28 = !DILocation(line: 13, column: 11, scope: !22)
!29 = !DILocation(line: 13, column: 8, scope: !22)
!30 = !DILocation(line: 14, column: 15, scope: !22)
!31 = !DILocation(line: 14, column: 11, scope: !22)
!32 = !DILocation(line: 14, column: 8, scope: !22)
!33 = !DILocation(line: 15, column: 18, scope: !22)
!34 = !DILocation(line: 15, column: 11, scope: !22)
!35 = !DILocation(line: 15, column: 8, scope: !22)
!36 = !DILocation(line: 16, column: 15, scope: !22)
!37 = !DILocation(line: 16, column: 11, scope: !22)
!38 = !DILocation(line: 16, column: 8, scope: !22)
!39 = !DILocation(line: 12, column: 37, scope: !22)
!40 = !DILocation(line: 12, column: 3, scope: !22)
!41 = distinct !{!41, !42, !43}
!42 = !DILocation(line: 12, column: 3, scope: !22)
!43 = !DILocation(line: 17, column: 3, scope: !22)
!44 = !DILocation(line: 18, column: 1, scope: !22)
