; Precommit test: a context-sensitive (CS) extbinary profile that has already
; been loaded is decoded again, and merged into itself, when the sample loader
; performs incremental reads (even without a symbol remapping file).
;
; The profile contains [_ZN2ns3barEi] and its child context
; [_ZN2ns3barEi:1 @ _Z3fooP4Basei]. The module defines _Z3fooP4Basei, so the
; initial Reader->read() in SampleProfileLoader::doInitialization loads
; [_ZN2ns3barEi:1 @ _Z3fooP4Basei] (100 samples on line 1 and line 3).
;
; _ZN2ns3barEl has no profile, but ns::bar(int) in the profile shares its base
; name, so -salvage-unused-profile issues incremental Reader.read() calls for
; the single candidate _ZN2ns3barEi. foo is not part of that request.
; Nevertheless:
;
;  1) In matchFunctionsWithoutProfileByBasename, Reader.read({"_ZN2ns3barEi"})
;     matches [_ZN2ns3barEi] in FuncOffsetList, sets CommonContext to
;     &[_ZN2ns3barEi], and walks all entries sharing that context prefix.
;     Because CommonContext->isPrefixOf([_ZN2ns3barEi:1 @ _Z3fooP4Basei]) is
;     true, the already-loaded child context for foo is decoded a second time
;     and accumulated into itself (200 samples).
;  2) In UpdateWithSalvagedProfiles, Reader.read({"_ZN2ns3barEi"}) is called
;     again. SampleProfileReader::read() pre-filters FuncsToUse with
;     Profiles.find(FunctionId(F)), which probes a context-less SampleContext
;     (hasContext() == false), whereas CS entries in Profiles have
;     hasContext() == true. The pre-filter misses, readFuncProfiles runs a
;     second incremental time, and [_ZN2ns3barEi:1 @ _Z3fooP4Basei] is decoded
;     a third time (300 samples).
;
; RUN: llvm-profdata merge -sample -extbinary %S/Inputs/cs-reread-loaded-profile.prof -o %t.cs.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.cs.afdo -salvage-stale-profile -salvage-unused-profile -S | FileCheck %s

; CHECK: define {{.*}} @_Z3fooP4Basei({{.*}} !prof ![[FOO_ENTRY:[0-9]+]]
; CHECK: br i1 %cmp, label %if.then, label %if.end, {{.*}} !prof ![[FOO_BR:[0-9]+]]
; CHECK: define {{.*}} @_ZN2ns3barEl({{.*}} !prof ![[BAR_ENTRY:[0-9]+]]

; FIXME: [_ZN2ns3barEi:1 @ _Z3fooP4Basei] is read three times (initial read +
; two incremental reads for _ZN2ns3barEi) and merged into itself; its samples
; are tripled. These should be 100 and "100, 0".
; CHECK-DAG: ![[FOO_ENTRY]] = !{!"function_entry_count", i64 300}
; CHECK-DAG: ![[FOO_BR]] = !{!"branch_weights", i32 300, i32 0}
; CHECK-DAG: ![[BAR_ENTRY]] = !{!"function_entry_count", i64 -1}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3fooP4Basei(ptr %b, i32 %x) #0 !dbg !10 {
entry:
  %cmp = icmp sgt i32 %x, 0, !dbg !12
  br i1 %cmp, label %if.then, label %if.end, !dbg !12

if.then:
  %vtable = load ptr, ptr %b, align 8, !dbg !13
  %fp = load ptr, ptr %vtable, align 8, !dbg !13
  %r = call i32 %fp(ptr %b), !dbg !13
  br label %if.end, !dbg !13

if.end:
  %s = phi i32 [ %r, %if.then ], [ 0, %entry ], !dbg !14
  ret i32 %s, !dbg !14
}

define i32 @_ZN2ns3barEl(i64 %x) #0 !dbg !20 {
entry:
  %t = trunc i64 %x to i32, !dbg !22
  %c = call i32 @_Z4sinki(i32 %t), !dbg !22
  %a = add i32 %c, 1, !dbg !23
  ret i32 %a, !dbg !23
}

declare i32 @_Z4sinki(i32)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, debugInfoForProfiling: true)
!1 = !DIFile(filename: "cs-reread-loaded-profile.cc", directory: "/tmp")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !6)
!6 = !{}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooP4Basei", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DILocation(line: 3, column: 7, scope: !10)
!13 = !DILocation(line: 4, column: 10, scope: !15)
!14 = !DILocation(line: 5, column: 3, scope: !10)
!15 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 1)
!20 = distinct !DISubprogram(name: "bar", linkageName: "_ZN2ns3barEl", scope: !1, file: !1, line: 10, type: !5, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!22 = !DILocation(line: 11, column: 10, scope: !24)
!23 = !DILocation(line: 11, column: 3, scope: !20)
!24 = !DILexicalBlockFile(scope: !20, file: !1, discriminator: 1)
