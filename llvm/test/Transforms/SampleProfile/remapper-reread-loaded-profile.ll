; Precommit test: an extbinary profile that has already been loaded is decoded a
; second time, and merged into itself, when the sample loader performs an
; incremental read while a symbol remapping file is in use.
;
; The module defines _Z3fooP4Basei, which has an exact profile match and is
; loaded by the initial Reader->read() in SampleProfileLoader::doInitialization.
; _ZN2ns3barEl has no profile, but ns::bar(int) in the profile shares its base
; name, so -salvage-unused-profile triggers an incremental Reader.read() for the
; single candidate _ZN2ns3barEi. foo is not part of that request. Nevertheless:
;
;  * Without a remapping file, readFuncProfiles looks up only the requested
;    names, foo is untouched, and its entry count is head samples + 1 = 101.
;  * With a remapping file, readFuncProfiles iterates over every function in the
;    profile and also accepts names for which Remapper->exist() is true. The
;    remapper is cumulative across read() calls, foo was inserted by the first
;    read, so foo is decoded again and its head, body and callsite samples are
;    doubled: entry count 201. The remapping file's content is irrelevant; the
;    rule in it matches nothing in this test.
;
; bar(int) is read exactly once in both cases (51).
;
; RUN: llvm-profdata merge -sample -extbinary %S/Inputs/remapper-reread-loaded-profile.prof -o %t.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.afdo -salvage-stale-profile -salvage-unused-profile -S | FileCheck %s --check-prefixes=CHECK,NOREMAP
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.afdo -sample-profile-remapping-file=%S/Inputs/remapper-reread-loaded-profile.map -salvage-stale-profile -salvage-unused-profile -S | FileCheck %s --check-prefixes=CHECK,REMAP

; CHECK: define {{.*}} @_Z3fooP4Basei({{.*}} !prof ![[FOO_ENTRY:[0-9]+]]
; CHECK: define {{.*}} @_ZN2ns3barEl({{.*}} !prof ![[BAR_ENTRY:[0-9]+]]

; NOREMAP-DAG: ![[FOO_ENTRY]] = !{!"function_entry_count", i64 101}
; FIXME: foo's profile is read twice and merged into itself; this should be 101.
; REMAP-DAG:   ![[FOO_ENTRY]] = !{!"function_entry_count", i64 201}
; CHECK-DAG:   ![[BAR_ENTRY]] = !{!"function_entry_count", i64 51}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3fooP4Basei(ptr %b, i32 %x) #0 !dbg !10 {
entry:
  %vtable = load ptr, ptr %b, align 8, !dbg !13
  %fp = load ptr, ptr %vtable, align 8, !dbg !13
  %r = call i32 %fp(ptr %b), !dbg !13
  ret i32 %r, !dbg !14
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
!1 = !DIFile(filename: "remapper-reread-loaded-profile.cc", directory: "/tmp")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !6)
!6 = !{}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooP4Basei", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 4, column: 10, scope: !15)
!14 = !DILocation(line: 5, column: 3, scope: !10)
!15 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 1)
!20 = distinct !DISubprogram(name: "bar", linkageName: "_ZN2ns3barEl", scope: !1, file: !1, line: 10, type: !5, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!22 = !DILocation(line: 11, column: 10, scope: !24)
!23 = !DILocation(line: 11, column: 3, scope: !20)
!24 = !DILexicalBlockFile(scope: !20, file: !1, discriminator: 1)
