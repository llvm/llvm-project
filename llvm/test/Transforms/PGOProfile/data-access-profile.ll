; REQUIRES: asserts
; asserts are required for -debug-only=<pass-name> and -stats

; RUN: rm -rf %t && split-file %s %t && cd %t

;; Read text profiles and merge them into indexed profiles.
; RUN: llvm-profdata merge --memprof-version=4 memprof.yaml -o memprof.profdata
; RUN: llvm-profdata merge --memprof-version=4 memprof-no-dap.yaml -o memprof-no-dap.profdata

;; Run optimizer pass on an IR module without IR functions, and test that global
;; variables in the module could be annotated (i.e., no early return),
; RUN: opt -passes='memprof-use<profile-filename=memprof.profdata>' -memprof-annotate-static-data-prefix \
; RUN: -debug-only=memprof -stats -S funcless-module.ll -o - 2>&1 | FileCheck %s --check-prefixes=LOGCOMMON,IRCOMMON,IR,STAT

;; Add '-memprof-annotate-string-literal-section-prefix' to RUN command above.
; RUN: opt -passes='memprof-use<profile-filename=memprof.profdata>' -memprof-annotate-static-data-prefix \
; RUN: -memprof-annotate-string-literal-section-prefix \
; RUN: -debug-only=memprof -stats -S funcless-module.ll -o - 2>&1 | FileCheck %s --check-prefixes=LOGCOMMON,LOGSTR,IRCOMMON,IRSTR,STRSTAT

;; Run optimizer pass on the IR, and check the section prefix.
; RUN: opt -passes='memprof-use<profile-filename=memprof.profdata>' -memprof-annotate-static-data-prefix \
; RUN: -debug-only=memprof -stats -S input.ll -o - 2>&1 | FileCheck %s --check-prefixes=LOGCOMMON,IR,STAT

;; Add '-memprof-annotate-string-literal-section-prefix' to RUN command above.
; RUN: opt -passes='memprof-use<profile-filename=memprof.profdata>' -memprof-annotate-static-data-prefix \
; RUN: -memprof-annotate-string-literal-section-prefix \
; RUN: -debug-only=memprof -stats -S input.ll -o - 2>&1 | FileCheck %s --check-prefixes=LOGCOMMON,LOGSTR,IRSTR,STRSTAT

;; Run memprof without providing memprof data. Test that IR has module flag
;; `EnableDataAccessProf` as 0.
; RUN: opt -passes='memprof-use<profile-filename=memprof-no-dap.profdata>' -memprof-annotate-static-data-prefix \
; RUN: -debug-only=memprof -stats -S input.ll -o - 2>&1 | FileCheck %s --check-prefix=FLAG

;; Run memprof without explicitly setting -memprof-annotate-static-data-prefix.
;; The output text IR shouldn't have `section_prefix` or EnableDataAccessProf module flag.
; RUN: opt -passes='memprof-use<profile-filename=memprof.profdata>' \
; RUN: -debug-only=memprof -stats -S input.ll -o - | FileCheck %s --check-prefix=FLAGLESS --implicit-check-not="section_prefix"

; LOGSTR: Global variable .str is annotated as hot

;; Common log lines
; LOGCOMMON: Global variable var1 is annotated as hot
; LOGCOMMON: Global variable var2.llvm.125 is annotated as hot
; LOGCOMMON: Global variable bar is not annotated
; LOGCOMMON: Global variable foo is annotated as unlikely
; LOGCOMMON: Skip annotation for var3 due to explicit section name.
; LOGCOMMON: Skip annotation for var4 due to explicit section name.
; LOGCOMMON: Skip annotation for llvm.fake_var due to name starts with `llvm.`.
; LOGCOMMON: Skip annotation for qux due to linker declaration.

; LOGSTR: Global variable .str.llvm.98765 is annotated as unlikely
; LOGSTR: Global variable .str.2 is not annotated

;; String literals are not annotated.
; IR: @.str = unnamed_addr constant [5 x i8] c"abcde"
; IR-NOT: section_prefix
; IR-SAME: {{$}}

;; .str is hot
; IRSTR: @.str = unnamed_addr constant [5 x i8] c"abcde", !section_prefix !0

; IRCOMMON: @var1 = global i32 123, !section_prefix !0

;; @var.llvm.125 will be canonicalized to @var2 for profile look-up.
; IRCOMMON-NEXT: @var2.llvm.125 = global i64 0, !section_prefix !0

;; @bar is not seen in hot symbol or known symbol set, so it won't get a section
;; prefix. Test this by testing that there is no section_prefix between @bar and
;; @foo.
; IRCOMMON-NEXT: @bar = global i16 3
; IRCOMMON-NOT: !section_prefix
; IRCOMMON-SAME: {{$}}

;; @foo is unlikely.
; IRCOMMON-NEXT: @foo = global i8 2, !section_prefix !1

; IRCOMMON-NEXT: @var3 = constant [2 x i32] [i32 12345, i32 6789], section "sec1"
; IRCOMMON-NEXT: @var4 = constant [1 x i64] [i64 98765] #0

; IRCOMMON: @llvm.fake_var = global i32 123
; IRCOMMON-NOT: !section_prefix
; IRCOMMON-SAME: {{$}}
; IRCOMMON: @qux = external global i64
; IRCOMMON-NOT: !section_prefix
; IRCOMMON-SAME: {{$}}

;; @.str.llvm.98765 is unlikely and @.str.2 has no section prefix.
; IRSTR: @.str.llvm.98765 = constant [5 x i8] c"Joins", align 1, !section_prefix !1
; IRSTR: @.str.2 = constant [15 x i8] c"*ptr == nullptr", align 1
; IRSTR-NOT: section_prefix
; IRSTR-SAME: {{$}}

; IR: attributes #0 = { "rodata-section"="sec2" }

; IR: !0 = !{!"section_prefix", !"hot"}
; IR-NEXT: !1 = !{!"section_prefix", !"unlikely"}
; IR-NEXT: !2 = !{i32 2, !"EnableDataAccessProf", i32 1}

; FLAG: !{i32 2, !"EnableDataAccessProf", i32 0}
; FLAGLESS-NOT: EnableDataAccessProf

; STAT: 1 memprof - Number of global vars annotated with 'unlikely' section prefix.
; STAT: 2 memprof - Number of global vars with user-specified section (not annotated).
; STAT: 2 memprof - Number of global vars annotated with 'hot' section prefix.
; STAT: 1 memprof - Number of global vars with unknown hotness (no section prefix).

; STRSTAT: 2 memprof - Number of global vars annotated with 'unlikely' section prefix.
; STRSTAT: 2 memprof - Number of global vars with user-specified section (not annotated).
; STRSTAT: 3 memprof - Number of global vars annotated with 'hot' section prefix.
; STRSTAT: 2 memprof - Number of global vars with unknown hotness (no section prefix).

;--- memprof.yaml
---
DataAccessProfiles:
  SampledRecords:
    - Symbol:          var1
      AccessCount:     1000
    - Symbol:          var2
      AccessCount:     5
      # 4211217683648370347 is the hash for 'abcde'
    - Hash:            4211217683648370347
      AccessCount:     145
  KnownColdSymbols:
    - foo
  # 1698928196410683810 is the hash for 'Joins'
  # 9184168289490324740 is the hash for 'ptr = nullptr'. This string doesn't exist in the IR.
  KnownColdStrHashes: [ 1698928196410683810, 9184168289490324740 ]
...
;--- memprof-no-dap.yaml
---
# A memprof file with without data access profiles. The heap records are simplified
# to pass profile parsing and don't need to match the IR.
HeapProfileRecords:
  - GUID:            0xdeadbeef12345678
    AllocSites:
      - Callstack:
          - { Function: 0x1111111111111111, LineOffset: 11, Column: 10, IsInlineFrame: true }
        MemInfoBlock:
          AllocCount:      111
          TotalSize:       222
          TotalLifetime:   333
          TotalLifetimeAccessDensity: 444
    CallSites:
      - Frames:
        - { Function: 0x5555555555555555, LineOffset: 55, Column: 50, IsInlineFrame: true }
...
;--- input.ll

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = unnamed_addr constant [5 x i8] c"abcde"
@var1 = global i32 123
@var2.llvm.125 = global i64 0
@bar = global i16 3
@foo = global i8 2
@var3 = constant [2 x i32][i32 12345, i32 6789], section "sec1"
@var4 = constant [1 x i64][i64 98765] #0
@llvm.fake_var = global i32 123
@qux = external global i64
@.str.llvm.98765 = constant [5 x i8] c"Joins", align 1
@.str.2 = constant [15 x i8] c"*ptr == nullptr", align 1

define i32 @func() {
  %a = load i32, ptr @var1
  %b = load i32, ptr @var2.llvm.125
  %c = load i32, ptr @llvm.fake_var
  %ret = call i32 (...) @func_taking_arbitrary_param(i32 %a, i32 %b, i32 %c)
  ret i32 %ret
}

declare i32 @func_taking_arbitrary_param(...)

attributes #0 = { "rodata-section"="sec2" }

;--- funcless-module.ll

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = unnamed_addr constant [5 x i8] c"abcde"
@var1 = global i32 123
@var2.llvm.125 = global i64 0
@bar = global i16 3
@foo = global i8 2
@var3 = constant [2 x i32][i32 12345, i32 6789], section "sec1"
@var4 = constant [1 x i64][i64 98765] #0
@llvm.fake_var = global i32 123
@qux = external global i64
@.str.llvm.98765 = constant [5 x i8] c"Joins", align 1
@.str.2 = constant [15 x i8] c"*ptr == nullptr", align 1


attributes #0 = { "rodata-section"="sec2" }
