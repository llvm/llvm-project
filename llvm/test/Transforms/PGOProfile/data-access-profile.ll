; RUN: rm -rf %t && split-file %s %t && cd %t

;; Read a text profile and merge it into indexed profile.
; RUN: llvm-profdata merge --memprof-version=4 memprof.yaml -o memprof.profdata

;; Run optimizer pass on the IR, and check the section prefix.
; RUN: opt -passes='memprof-use<profile-filename=memprof.profdata>' -annotate-static-data-prefix \
; RUN: -S input.ll -o - 2>&1 | FileCheck %s

;; Repeat the command line above and enable -print-static-data-prefix. Test both IR and log.
; RUN: opt -passes='memprof-use<profile-filename=memprof.profdata>' -annotate-static-data-prefix \
; RUN: -print-static-data-prefix -S input.ll -o - 2>&1 | FileCheck %s --check-prefixes=LOG,CHECK

; LOG: Global variable .str has no section prefix
; LOG: Global variable var1 has section prefix: hot
; LOG: Global variable var2.llvm.125 has section prefix: hot
; LOG: Global variable foo has section prefix: unlikely
; LOG: Global variable bar has no section prefix

;; String literals are not annotated.
; CHECK: @.str = unnamed_addr constant [5 x i8] c"abcde"
; CHECK-NOT: section_prefix
; CHECK: @var1 = global i32 123, !section_prefix !0

;; @var.llvm.125 will be canonicalized to @var2 for profile look-up.
; CHECK-NEXT: @var2.llvm.125 = global i64 0, !section_prefix !0
; CHECK-NEXT: @foo = global i8 2, !section_prefix !1

;; @bar is not seen in hot symbol or known symbol set, so it doesn't get
;; a section prefix. It's up to the linker to decide how to map input sections
;; to output, and one conservative practice is to map unlikely-prefixed ones to
;; unlikely output section, and map the rest (hot-prefixed or prefix-less) to
;; the canonical output section.
; CHECK-NEXT: @bar = global i16 3

; CHECK: !0 = !{!"section_prefix", !"hot"}
; CHECK-NEXT: !1 = !{!"section_prefix", !"unlikely"}

;--- memprof.yaml
---
HeapProfileRecords:
  - GUID:            0xdeadbeef12345678
    AllocSites:
      - Callstack:
          - { Function: 0x1111111111111111, LineOffset: 11, Column: 10, IsInlineFrame: true }
          - { Function: 0x2222222222222222, LineOffset: 22, Column: 20, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      111
          TotalSize:       222
          TotalLifetime:   333
          TotalLifetimeAccessDensity: 444
    CallSites:
      - Frames:
        - { Function: 0x5555555555555555, LineOffset: 55, Column: 50, IsInlineFrame: true }
        - { Function: 0x6666666666666666, LineOffset: 66, Column: 60, IsInlineFrame: false }
        CalleeGuids: [ 0x100, 0x200 ]
DataAccessProfiles:
  SampledRecords:
    - Symbol:          var1
      AccessCount:     1000
    - Symbol:          var2
      AccessCount:     5
    - Hash:            101010
      AccessCount:     145
  KnownColdSymbols:
    - foo
  KnownColdStrHashes: [ 999, 1001 ]
...
;--- input.ll

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = unnamed_addr constant [5 x i8] c"abcde"
@var1 = global i32 123
@var2.llvm.125 = global i64 0 
@foo = global i8 2
@bar = global i16 3

define i32 @func() {
  %a = load i32, ptr @var1
  %b = load i32, ptr @var2.llvm.125
  %ret = call i32 (...) @func_taking_arbitrary_param(i32 %a, i32 %b)
  ret i32 %ret
}

declare i32 @func_taking_arbitrary_param(...)
