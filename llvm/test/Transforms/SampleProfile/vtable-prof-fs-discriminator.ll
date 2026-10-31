;; With FS discriminators, distinct raw profile locations (from debug info) can
;; map to the same masked location. Body and callsite samples are accumulated in
;; that case; test that vtable type samples are accumulated the same way, and
;; that there are no warnings of 'Duplicate vtable type'.

;; In vtable-prof-fs-discriminator.prof, 3.1 and 3.257, 5.1 and 5.257 collapse
;; to one location ( 257 = 1 | 1 << 8 ).
;
; RUN: llvm-profdata merge -sample -extbinary -extbinary-write-vtable-type-prof \
; RUN: -profile-isfs %S/Inputs/vtable-prof-fs-discriminator.prof -o %t.fs.afdo
;
; Reading the profile back with every discriminator bit.
; RUN: llvm-profdata merge -sample -text -fs-discriminator-pass=PassLast \
; RUN: %t.fs.afdo -o - 2>&1 | FileCheck %s --check-prefix=RAW --match-full-lines \
; RUN: --implicit-check-not='Duplicate vtable type'
;
; Reading it back with the base mask, this is what the IR-level sample loader
; sees.
; RUN: llvm-profdata merge -sample -text -fs-discriminator-pass=Base \
; RUN: %t.fs.afdo -o - 2>&1 | FileCheck %s --check-prefix=MERGED --match-full-lines \
; RUN: --implicit-check-not='Duplicate vtable type'
;
; The sample loader itself consumes the merged profile without a diagnostic.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.fs.afdo -S 2>&1 \
; RUN: | FileCheck %s --implicit-check-not='Duplicate vtable type'

; RAW:       _Z3fooP4Basei:1000:100
; RAW-NEXT:   1: 100
; RAW-NEXT:   3: 100
; RAW-NEXT:   3.1: 900 _ZN4Base1fEv:900
; RAW-NEXT:   3.1: vtables _ZTV4Base:900
; RAW-NEXT:   3.257: 100 _ZN4Base1fEv:100
; RAW-NEXT:   3.257: vtables _ZTV4Base:100
; RAW-NEXT:   4: 100
; RAW-NEXT:   5.1: _Z3barv:70
; RAW-NEXT:    2: 70
; RAW-NEXT:   5.257: _Z3barv:40
; RAW-NEXT:    2: 40

;; Only base discriminators (lower 8 bits) are retained, and higher bits are
;; masked.
; MERGED:       _Z3fooP4Basei:1000:100
; MERGED-NEXT:   1: 100
; MERGED-NEXT:   3: 100
; MERGED-NEXT:   3.1: 1000 _ZN4Base1fEv:1000
; MERGED-NEXT:   3.1: vtables _ZTV4Base:1000
; MERGED-NEXT:   4: 100
; MERGED-NEXT:   5.1: _Z3barv:110
; MERGED-NEXT:    2: 110

; CHECK: define {{.*}} @_Z3fooP4Basei({{.*}} !prof ![[FOO_ENTRY:[0-9]+]]
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
