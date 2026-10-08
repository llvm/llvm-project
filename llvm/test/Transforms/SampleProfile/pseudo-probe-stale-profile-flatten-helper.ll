; Test that functionMatchesProfileHelper flattens profiles after loading.
;
; When functionMatchesProfileHelper loads a top-level profile via
; Reader.read(), it must flatten the profile so inlined callees get their
; own entries in FlattenedProfiles. Without flattening, a renamed callee
; that only exists as an inlined entry (not top-level) cannot be matched
; because its profile is undiscoverable.
;
; Setup:
;   - main calls foo_new (renamed from foo_old, matched by checksum)
;   - foo_new calls bar_new (renamed from bar_old)
;   - bar_old exists ONLY as an inlined callee in foo_old's profile
;   - Non-C++ names so basename matching fails; matching uses checksums
;   - Without flattening: bar_old has no flattened entry, bar_new unmatched
;   - With flattening: bar_old gets flattened, bar_new matched by checksum

; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-stale-profile-flatten-helper.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl --min-func-count-for-cg-matching=0 --min-call-count-for-cg-matching=0 2>&1 | FileCheck %s

; Verify orphan functions are found.
; CHECK: Function bar_new is not in profile or profile symbol list.
; CHECK: Function foo_new is not in profile or profile symbol list.

; Verify foo_new is matched to foo_old by checksum via CG matching.
; CHECK: The checksums for foo_new(IR) and foo_old(Profile) match.
; CHECK: Function:foo_new matches profile:foo_old

; Verify bar_new is matched to bar_old by checksum. This requires
; foo_old's profile to be flattened after loading so bar_old gets a
; flattened entry discoverable by getFlattenedSamplesFor.
; CHECK: The checksums for bar_new(IR) and bar_old(Profile) match.
; CHECK: Function:bar_new matches profile:bar_old

; Verify stale profile matching runs for bar_new (would not happen
; without flattening because bar_old's profile would be unavailable).
; CHECK: Run stale profile matching for bar_new


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @bar_new() #0 !dbg !20 {
entry:
  call void @llvm.pseudoprobe(i64 8236371237083957767, i64 1, i32 0, i64 -1), !dbg !23
  ret void, !dbg !24
}

define void @foo_new() #0 !dbg !11 {
entry:
  call void @llvm.pseudoprobe(i64 -837213161392124280, i64 1, i32 0, i64 -1), !dbg !14
  call void @bar_new(), !dbg !40
  ret void, !dbg !17
}

define i32 @main() #0 !dbg !30 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !33
  call void @foo_new(), !dbg !42
  ret i32 0, !dbg !36
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { noinline nounwind "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!9, !10, !29}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"uwtable", i32 2}
!9 = !{i64 -837213161392124280, i64 222222, !"foo_new"}
!10 = !{i64 8236371237083957767, i64 333333, !"bar_new"}
!29 = !{i64 -2624081020897602054, i64 111111, !"main"}

!11 = distinct !DISubprogram(name: "foo_new", linkageName: "foo_new", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
!14 = !DILocation(line: 4, scope: !11)
!17 = !DILocation(line: 6, scope: !11)

!20 = distinct !DISubprogram(name: "bar_new", linkageName: "bar_new", scope: !1, file: !1, line: 10, type: !12, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!23 = !DILocation(line: 11, scope: !20)
!24 = !DILocation(line: 12, scope: !20)

!30 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !1, file: !1, line: 20, type: !12, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!33 = !DILocation(line: 21, scope: !30)
!36 = !DILocation(line: 24, scope: !30)

; Call probe discriminators: (ProbeIndex << 3) | (100 << 19) | (2 << 26) | 0x7
!41 = !DILexicalBlockFile(scope: !30, file: !1, discriminator: 186646575)
!42 = !DILocation(line: 23, scope: !41)
!39 = !DILexicalBlockFile(scope: !11, file: !1, discriminator: 186646559)
!40 = !DILocation(line: 5, scope: !39)
