; With flow-sensitive discriminators, the IR-level sample loader masks profile
; discriminators down to the base discriminator. Two raw profile locations
; (3.1 and 3.257 below) therefore collapse to one location; their vtable type
; counters must be accumulated like body/callsite samples are, not reported as
; duplicates.
;
; RUN: llvm-profdata merge -sample -extbinary -extbinary-write-vtable-type-prof -profile-isfs %S/Inputs/vtable-prof-fs-discriminator.prof -o %t.fs.afdo
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.fs.afdo -S 2>&1 | FileCheck %s

; CHECK-NOT: Duplicate vtable type
; CHECK: define {{.*}} @_Z3fooP4Basei({{.*}} !prof ![[FOO_ENTRY:[0-9]+]]
; CHECK-NOT: Duplicate vtable type
; CHECK: ![[FOO_ENTRY]] = !{!"function_entry_count", i64 101}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3fooP4Basei(ptr %b, i32 %x) #0 !dbg !10 {
entry:
  %vtable = load ptr, ptr %b, align 8, !dbg !13
  %fp = load ptr, ptr %vtable, align 8, !dbg !13
  %r = call i32 %fp(ptr %b), !dbg !13
  ret i32 %r, !dbg !14
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, debugInfoForProfiling: true)
!1 = !DIFile(filename: "vtable-prof-fs-discriminator.cc", directory: "/tmp")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !6)
!6 = !{}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooP4Basei", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 4, column: 10, scope: !15)
!14 = !DILocation(line: 5, column: 3, scope: !10)
!15 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 1)
