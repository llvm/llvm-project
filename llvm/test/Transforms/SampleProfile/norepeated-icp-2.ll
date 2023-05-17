; RUN: opt < %s -sample-profile-icp-max-prom=2 -passes=sample-profile -sample-profile-file=%S/Inputs/norepeated-icp-2.prof -S | FileCheck %s --check-prefix=MAX2
; RUN: opt < %s -sample-profile-icp-max-prom=4 -passes=sample-profile -sample-profile-file=%S/Inputs/norepeated-icp-2.prof -S | FileCheck %s --check-prefix=MAX4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"moo\0A\00", align 1
@p = dso_local global ptr null, align 8
@cond = dso_local global i8 0, align 1
@str = private unnamed_addr constant [4 x i8] c"moo\00", align 1

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3moov() #0 !dbg !7 {
entry:
  %puts = call i32 @puts(ptr nonnull dereferenceable(1) @str), !dbg !9
  ret void, !dbg !10
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) #1

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3hoov() #0 !dbg !11 {
entry:
  %0 = load volatile i8, ptr @cond, align 1, !dbg !12, !range !17
  %tobool.not = icmp eq i8 %0, 0, !dbg !12
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !12

if.then:                                          ; preds = %entry
  call void @_Z10hoo_calleev(), !dbg !18
  br label %if.end, !dbg !18

if.end:                                           ; preds = %if.then, %entry
  store ptr @_Z3moov, ptr @p, align 8, !dbg !19
  ret void, !dbg !22
}

declare !dbg !23 dso_local void @_Z10hoo_calleev() #2

; MAX2-LABEL: @_Z3goov(
; MAX2: icmp eq ptr {{.*}} @_Z3hoov
; MAX2: call void %t0(), {{.*}} !prof ![[PROF_ID1:[0-9]+]]
; MAX2-NOT: icmp eq ptr {{.*}} @_Z3hoov
; MAX2-NOT: icmp eq ptr {{.*}} @_Z3moov
; MAX2: call void %t1(), {{.*}} !prof ![[PROF_ID2:[0-9]+]]
; MAX2-NOT: icmp eq ptr {{.*}} @_Z3hoov
; MAX2-NOT: icmp eq ptr {{.*}} @_Z3moov
; MAX2: call void %t2(), {{.*}} !prof ![[PROF_ID2:[0-9]+]]
; MAX2: ret void
; MAX4-LABEL: @_Z3goov(
; MAX4: icmp eq ptr {{.*}} @_Z3hoov
; MAX4: icmp eq ptr {{.*}} @_Z3moov
; MAX4: call void %t0(), {{.*}} !prof ![[PROF_ID3:[0-9]+]]
; MAX4: icmp eq ptr {{.*}} @_Z3hoov
; MAX4: icmp eq ptr {{.*}} @_Z3moov
; MAX4: call void %t1(), {{.*}} !prof ![[PROF_ID4:[0-9]+]]
; MAX4-NOT: icmp eq ptr {{.*}} @_Z3hoov
; MAX4-NOT: icmp eq ptr {{.*}} @_Z3moov
; MAX4: call void %t2(), {{.*}} !prof ![[PROF_ID5:[0-9]+]]
; MAX4: ret void

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3goov() #0 !dbg !24 {
entry:
  %t0 = load ptr, ptr @p, align 8, !dbg !25
  call void %t0(), !dbg !26, !prof !30
  %t1 = load ptr, ptr @p, align 8, !dbg !25
  call void %t1(), !dbg !28, !prof !31
  %t2 = load ptr, ptr @p, align 8, !dbg !25
  call void %t2(), !dbg !29, !prof !32
  ret void, !dbg !27
}

; MAX2: ![[PROF_ID1]] = !{!"VP", i32 0, i64 13000, i64 -7701940972712279918, i64 -1, i64 1850239051784516332, i64 -1}
; MAX2: ![[PROF_ID2]] = !{!"VP", i32 0, i64 13000, i64 3137940972712279918, i64 -1, i64 1850239051784516332, i64 -1}
; MAX4: ![[PROF_ID3]] = !{!"VP", i32 0, i64 13000, i64 -7383239051784516332, i64 -1, i64 -7701940972712279918, i64 -1, i64 1850239051784516332, i64 -1, i64 9191153033785521275, i64 2000}
; MAX4: ![[PROF_ID4]] = !{!"VP", i32 0, i64 13000, i64 -7383239051784516332, i64 -1, i64 -7701940972712279918, i64 -1, i64 3137940972712279918, i64 -1, i64 1850239051784516332, i64 -1}
; MAX4: ![[PROF_ID5]] = !{!"VP", i32 0, i64 13000, i64 4128940972712279918, i64 -1, i64 3137940972712279918, i64 -1, i64 2132940972712279918, i64 -1, i64 1850239051784516332, i64 -1}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) #3

attributes #0 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-sample-profile" "use-soft-float"="false" }
attributes #1 = { nofree nounwind "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git f8226e6e284e9f199790bdb330f87d71adb5376f)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "1.cc", directory: "/usr/local/google/home/wmi/workarea/llvm/build/splitprofile")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git f8226e6e284e9f199790bdb330f87d71adb5376f)"}
!7 = distinct !DISubprogram(name: "moo", linkageName: "_Z3moov", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 3, scope: !7)
!10 = !DILocation(line: 3, column: 1, scope: !7)
!11 = distinct !DISubprogram(name: "hoo", linkageName: "_Z3hoov", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 10, column: 7, scope: !11)
!13 = !{!14, !14, i64 0}
!14 = !{!"bool", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}
!17 = !{i8 0, i8 2}
!18 = !DILocation(line: 11, column: 5, scope: !11)
!19 = !DILocation(line: 12, column: 5, scope: !11)
!20 = !{!21, !21, i64 0}
!21 = !{!"any pointer", !15, i64 0}
!22 = !DILocation(line: 13, column: 1, scope: !11)
!23 = !DISubprogram(name: "hoo_callee", linkageName: "_Z10hoo_calleev", scope: !1, file: !1, line: 5, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!24 = distinct !DISubprogram(name: "goo", linkageName: "_Z3goov", scope: !1, file: !1, line: 15, type: !8, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!25 = !DILocation(line: 16, column: 5, scope: !24)
!26 = !DILocation(line: 16, column: 3, scope: !24)
!27 = !DILocation(line: 19, column: 1, scope: !24)
!28 = !DILocation(line: 17, column: 3, scope: !24)
!29 = !DILocation(line: 18, column: 3, scope: !24)
!30 = !{!"VP", i32 0, i64 0, i64 1850239051784516332, i64 -1}
!31 = !{!"VP", i32 0, i64 0, i64 1850239051784516332, i64 -1, i64 3137940972712279918, i64 -1}
!32 = !{!"VP", i32 0, i64 0, i64 1850239051784516332, i64 -1, i64 3137940972712279918, i64 -1, i64 2132940972712279918, i64 -1, i64 4128940972712279918, i64 -1}
