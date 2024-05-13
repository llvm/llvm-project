; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/norepeated-icp.prof -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"hoo\0A\00", align 1
@p = dso_local global ptr null, align 8
@str = private unnamed_addr constant [4 x i8] c"hoo\00", align 1

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3hoov() #0 !dbg !7 {
entry:
  %puts = call i32 @puts(ptr nonnull dereferenceable(1) @str), !dbg !9
  ret void, !dbg !10
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) #1

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3goov() #0 !dbg !11 {
entry:
  %0 = load ptr, ptr @p, align 8, !dbg !12, !tbaa !13
  call void %0(), !dbg !17
  ret void, !dbg !18
}

; Check the indirect call in _Z3goov inlined into _Z3foov won't be indirect
; call promoted for _Z3hoov twice in _Z3foov.
; CHECK-LABEL: @_Z3foov(
; CHECK: icmp eq ptr {{.*}} @_Z3hoov
; CHECK-NOT: icmp eq ptr {{.*}} @_Z3hoov
; CHECK: ret void

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3foov() #0 !dbg !19 {
entry:
  call void @_Z3goov(), !dbg !20
  ret void, !dbg !21
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) #2

attributes #0 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-sample-profile" "use-soft-float"="false" }
attributes #1 = { nofree nounwind "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nofree nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "1.cc", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "hoo", linkageName: "_Z3hoov", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 3, scope: !7)
!10 = !DILocation(line: 3, column: 1, scope: !7)
!11 = distinct !DISubprogram(name: "goo", linkageName: "_Z3goov", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 7, column: 5, scope: !11)
!13 = !{!14, !14, i64 0}
!14 = !{!"any pointer", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}
!17 = !DILocation(line: 7, column: 3, scope: !11)
!18 = !DILocation(line: 8, column: 1, scope: !11)
!19 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 11, column: 3, scope: !19)
!21 = !DILocation(line: 12, column: 3, scope: !19)
