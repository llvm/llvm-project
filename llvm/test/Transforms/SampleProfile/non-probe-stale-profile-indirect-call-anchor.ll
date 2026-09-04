; REQUIRES: x86-registered-target
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/non-probe-stale-profile-indirect-call-anchor.prof --salvage-stale-profile -S | FileCheck %s

; An IR indirect call has no statically known callee, so its anchor is the dummy
; name UnknownIndirectCallee. Such an anchor must never be treated as matching an
; arbitrary profiled callee during the longest common subsequence: it carries no
; callee identity, so using it as an alignment landmark can shift the location
; mapping of the neighboring callsites.

; The profiled source code:
;
;   void test(int c) {
;     foo();       // line offset 1
;     bar();       // line offset 2
;   }
;
; The source code for the current build, with an indirect call inserted in front:
;
;   void test(int c) {
;     fp();        // line offset 1, code change
;     if (c)
;       foo();     // line offset 2
;     bar();       // line offset 3
;   }

; The only valid alignment anchors are foo and bar, giving 2->1 and 3->2, so the
; call to foo takes the 100 samples recorded at profile location 1. If the
; indirect call were allowed to anchor on location 1 instead, foo would slide onto
; location 2 and wrongly inherit bar's 800 samples.

; CHECK: tail call void @foo(), !dbg !{{[0-9]+}}, !prof ![[FOO:[0-9]+]]
; CHECK: ![[FOO]] = !{!"branch_weights", i32 100}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fp = dso_local local_unnamed_addr global ptr null, align 8

define dso_local void @test(i1 %c) local_unnamed_addr #0 !dbg !9 {
entry:
  %0 = load ptr, ptr @fp, align 8, !dbg !12
  tail call void %0(), !dbg !12
  br i1 %c, label %if.then, label %if.end, !dbg !12

if.then:
  tail call void @foo(), !dbg !13
  br label %if.end, !dbg !13

if.end:
  tail call void @bar(), !dbg !14
  ret void, !dbg !15
}

declare void @foo() local_unnamed_addr

declare void @bar() local_unnamed_addr

attributes #0 = { nounwind uwtable "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "test")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{!"clang"}
!9 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !10, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 6, column: 3, scope: !9)
!13 = !DILocation(line: 7, column: 3, scope: !9)
!14 = !DILocation(line: 8, column: 3, scope: !9)
!15 = !DILocation(line: 9, column: 1, scope: !9)
