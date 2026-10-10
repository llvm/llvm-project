; Check that an extbinary profile that is already loaded is not read a second
; time (and merged into itself) when the sample loader performs an incremental
; read for stale/unused profile matching while a remapping file is in use.
;
; _Z3fooP4Basei is in the module, so its profile is loaded by the initial read.
; _ZN2ns3barEl has no profile, but ns::bar(int) in the profile shares its base
; name, so -salvage-unused-profile triggers an incremental Reader.read(). With a
; remapper, names from the first read were still matched by the second one,
; which re-read foo's profile: head/body samples were doubled and the vtable
; counters were diagnosed as duplicates.
;
; RUN: llvm-profdata merge -sample -extbinary -extbinary-write-vtable-type-prof %S/Inputs/remap-incremental-read.prof -o %t.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.afdo -sample-profile-remapping-file=%S/Inputs/remap-incremental-read.map -salvage-stale-profile -salvage-unused-profile -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.afdo -salvage-stale-profile -salvage-unused-profile -S 2>&1 | FileCheck %s

; CHECK-NOT: Duplicate vtable type
; CHECK: define {{.*}} @_Z3fooP4Basei({{.*}} !prof ![[FOO_ENTRY:[0-9]+]]
; CHECK: define {{.*}} @_ZN2ns3barEl({{.*}} !prof ![[BAR_ENTRY:[0-9]+]]
; CHECK-NOT: Duplicate vtable type
; Head samples (100) + 1, not doubled to 201.
; CHECK-DAG: ![[FOO_ENTRY]] = !{!"function_entry_count", i64 101}
; CHECK-DAG: ![[BAR_ENTRY]] = !{!"function_entry_count", i64 51}

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
!1 = !DIFile(filename: "remap-incremental-read.cc", directory: "/tmp")
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
