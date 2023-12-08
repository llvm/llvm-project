; Make sure profile matching is successful if both profile and IR contain
; ".__uniq." suffix, for text format or extbinary format profile.
; Make sure profile matching is successful if IR contains ".__uniq." suffix
; but profile doesn't contain the suffix, for extbinary format profile.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/uniqname.suffix.prof -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/uniqname.suffix.afdo -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/uniqname.nosuffix.afdo -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@cond = dso_local global i8 0, align 1
@p = dso_local global ptr null, align 8

; Check the callsite in inlined function with uniq suffix is annotated with
; profile correctly.
; CHECK-LABEL: @_Z3foov(
; CHECK: call void @_Z10moo_calleev(), {{.*}} !prof ![[PROF_ID1:[0-9]+]]
; CHECK: call void @_Z10noo_calleev(), {{.*}} !prof ![[PROF_ID2:[0-9]+]]
; CHECK: ret void

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3foov() #0 !dbg !7 {
entry:
  store ptr @_ZL3hoov.__uniq.334154460836426447066042049082945760258, ptr @p, align 8, !dbg !9, !tbaa !10
  call void @_ZL3goov.__uniq.334154460836426447066042049082945760258.llvm.4206369970847378271(), !dbg !14
  call void @_ZL3moov.__uniq.334154460836426447066042049082945760258(), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: uwtable mustprogress
define internal void @_ZL3hoov.__uniq.334154460836426447066042049082945760258() #1 !dbg !17 {
entry:
  call void @_Z10hoo_calleev(), !dbg !18
  ret void, !dbg !19
}

; Check the indirect call target with uniq suffix is promoted and the inlined
; body is annotated with profile.
; CHECK: define internal void @_ZL3goov.__uniq.334154460836426447066042049082945760258.llvm.4206369970847378271{{.*}} !prof ![[PROF_ID3:[0-9]+]]
; CHECK: icmp eq ptr {{.*}} @_ZL3hoov.__uniq.334154460836426447066042049082945760258
; CHECK: call void @_Z10hoo_calleev(), {{.*}} !prof ![[PROF_ID4:[0-9]+]]

; Function Attrs: noinline uwtable mustprogress
define internal void @_ZL3goov.__uniq.334154460836426447066042049082945760258.llvm.4206369970847378271() #2 !dbg !20 {
entry:
  %0 = load ptr, ptr @p, align 8, !dbg !21, !tbaa !10
  call void %0(), !dbg !22
  ret void, !dbg !23
}

; Function Attrs: uwtable mustprogress
define internal void @_ZL3moov.__uniq.334154460836426447066042049082945760258() #1 !dbg !24 {
entry:
  call void @_Z10moo_calleev(), !dbg !25
  %0 = load volatile i8, ptr @cond, align 1, !dbg !26, !tbaa !27, !range !29
  %tobool.not = icmp eq i8 %0, 0, !dbg !26
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !26

if.then:                                          ; preds = %entry
  call void @_ZL3noov.__uniq.334154460836426447066042049082945760258(), !dbg !30
  br label %if.end, !dbg !30

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !31
}

declare !dbg !32 dso_local void @_Z10hoo_calleev() #3

declare !dbg !33 dso_local void @_Z10moo_calleev() #3

; Function Attrs: uwtable mustprogress
define internal void @_ZL3noov.__uniq.334154460836426447066042049082945760258() #1 !dbg !34 {
entry:
  %0 = load volatile i8, ptr @cond, align 1, !dbg !35, !tbaa !27, !range !29
  %tobool.not = icmp eq i8 %0, 0, !dbg !35
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !35

if.then:                                          ; preds = %entry
  call void @_Z10noo_calleev(), !dbg !36
  br label %if.end, !dbg !36

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !37
}

declare !dbg !38 dso_local void @_Z10noo_calleev() #3

attributes #0 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-sample-profile" "use-soft-float"="false" }
attributes #1 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "sample-profile-suffix-elision-policy"="selected" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-sample-profile" "use-soft-float"="false" }
attributes #2 = { noinline uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "sample-profile-suffix-elision-policy"="selected" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-sample-profile" "use-soft-float"="false" }
attributes #3 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

; CHECK: ![[PROF_ID1]] = !{!"branch_weights", i32 5931}
; CHECK: ![[PROF_ID2]] = !{!"branch_weights", i32 2000}
; CHECK: ![[PROF_ID3]] = !{!"function_entry_count", i64 5861}
; CHECK: ![[PROF_ID4]] = !{!"branch_weights", i32 5000}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git e42c17446a2e5cbf9eebc752beafadad3fac7f63)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "1.cc", directory: "/usr/local/google/home/wmi/workarea/llvm/build/splitprofile")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git e42c17446a2e5cbf9eebc752beafadad3fac7f63)"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 28, type: !8, scopeLine: 28, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 29, column: 5, scope: !7)
!10 = !{!11, !11, i64 0}
!11 = !{!"any pointer", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C++ TBAA"}
!14 = !DILocation(line: 30, column: 3, scope: !7)
!15 = !DILocation(line: 31, column: 3, scope: !7)
!16 = !DILocation(line: 32, column: 3, scope: !7)
!17 = distinct !DISubprogram(name: "hoo", linkageName: "_ZL3hoov.__uniq.334154460836426447066042049082945760258", scope: !1, file: !1, line: 17, type: !8, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 18, column: 3, scope: !17)
!19 = !DILocation(line: 19, column: 1, scope: !17)
!20 = distinct !DISubprogram(name: "goo", linkageName: "_ZL3goov.__uniq.334154460836426447066042049082945760258.llvm.4206369970847378271", scope: !1, file: !1, line: 24, type: !8, scopeLine: 24, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 25, column: 5, scope: !20)
!22 = !DILocation(line: 25, column: 3, scope: !20)
!23 = !DILocation(line: 26, column: 1, scope: !20)
!24 = distinct !DISubprogram(name: "moo", linkageName: "_ZL3moov.__uniq.334154460836426447066042049082945760258", scope: !1, file: !1, line: 11, type: !8, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!25 = !DILocation(line: 12, column: 3, scope: !24)
!26 = !DILocation(line: 13, column: 7, scope: !24)
!27 = !{!28, !28, i64 0}
!28 = !{!"bool", !12, i64 0}
!29 = !{i8 0, i8 2}
!30 = !DILocation(line: 14, column: 5, scope: !24)
!31 = !DILocation(line: 15, column: 1, scope: !24)
!32 = !DISubprogram(name: "hoo_callee", linkageName: "_Z10hoo_calleev", scope: !1, file: !1, line: 3, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!33 = !DISubprogram(name: "moo_callee", linkageName: "_Z10moo_calleev", scope: !1, file: !1, line: 2, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!34 = distinct !DISubprogram(name: "noo", linkageName: "_ZL3noov.__uniq.334154460836426447066042049082945760258", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!35 = !DILocation(line: 7, column: 7, scope: !34)
!36 = !DILocation(line: 8, column: 5, scope: !34)
!37 = !DILocation(line: 9, column: 1, scope: !34)
!38 = !DISubprogram(name: "noo_callee", linkageName: "_Z10noo_calleev", scope: !1, file: !1, line: 1, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
