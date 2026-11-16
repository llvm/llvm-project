;; Check ThinLTO post-link pass pipeline ordering for LowerTypeTestsPass,
;; WholeProgramDevirtPass, DropTypeTestsPass, SampleProfileLoaderPass, and
;; PGOIndirectCallPromotion when ImportSummary is non-null.
;; Note: `opt -passes='thinlto<O*>'` passes `ImportSummary = nullptr`, which
;; omits `WholeProgramDevirtPass` and `LowerTypeTestsPass`.

; RUN: rm -rf %t && split-file %s %t
; RUN: opt -module-summary %t/main.ll -o %t/main.bc
; RUN: llvm-ctxprof-util fromYAML --input=%t/ctxprof.yaml --output=%t/ctxprof.bitstream

; RUN: llvm-lto2 run %t/main.bc -o %t/o0.o -r=%t/main.bc,foo,plx \
; RUN:   -O0 -print-pipeline-passes \
; RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=lowertypetests

; RUN: llvm-lto2 run %t/main.bc -o %t/o1.o -r=%t/main.bc,foo,plx \
; RUN:   -O1 -print-pipeline-passes \
; RUN:   | FileCheck %s --check-prefix=OPT --implicit-check-not=lowertypetests
; RUN: llvm-lto2 run %t/main.bc -o %t/o2.o -r=%t/main.bc,foo,plx \
; RUN:   -O2 -print-pipeline-passes \
; RUN:   | FileCheck %s --check-prefix=OPT --implicit-check-not=lowertypetests
; RUN: llvm-lto2 run %t/main.bc -o %t/o3.o -r=%t/main.bc,foo,plx \
; RUN:   -O3 -print-pipeline-passes \
; RUN:   | FileCheck %s --check-prefix=OPT --implicit-check-not=lowertypetests

; RUN: llvm-lto2 run %t/main.bc -o %t/sample.o -r=%t/main.bc,foo,plx \
; RUN:   -O2 -lto-sample-profile-file=%t/sample.prof -print-pipeline-passes \
; RUN:   | FileCheck %s --check-prefix=SAMPLEPGO --implicit-check-not=lowertypetests

; RUN: llvm-lto2 run %t/main.bc -o %t/ctx.o -r=%t/main.bc,foo,plx \
; RUN:   -O2 -use-ctx-profile=%t/ctxprof.bitstream -print-pipeline-passes \
; RUN:   | FileCheck %s --check-prefix=CTXPROF --implicit-check-not=lowertypetests

; O0: pipeline-passes: {{.*}}wholeprogramdevirt,lowertypetests,drop-type-tests<assume>,

; OPT: pipeline-passes: {{.*}}wholeprogramdevirt,pgo-icall-prom,openmp-opt,lowertypetests,drop-type-tests<assume>,ipsccp,

; SAMPLEPGO: pipeline-passes: {{.*}}wholeprogramdevirt,sample-profile,require<profile-summary>,pgo-icall-prom,openmp-opt,lowertypetests,drop-type-tests<assume>,ipsccp,

; CTXPROF: pipeline-passes: {{.*}}wholeprogramdevirt,lowertypetests,module-inline,

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  ret void
}

;--- sample.prof
foo:100:100
 1: 100

;--- ctxprof.yaml
Contexts: []
