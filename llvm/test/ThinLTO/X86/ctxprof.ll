; Test workload based importing via -thinlto-pgo-ctx-prof
; Use external linkage symbols so we don't depend on module paths which are
; used when computing the GUIDs of internal linkage symbols.
; The functionality is shared with what workload.ll tests, so here we only care
; about testing the ctx profile is loaded and handled correctly.
;
; Set up
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: split-file %s %t
;
; RUN: opt -module-summary %t/m1.ll -o %t/m1.bc
; RUN: opt -module-summary %t/m2.ll -o %t/m2.bc
; RUN: llvm-dis %t/m1.bc -o - | FileCheck %s --check-prefix=GUIDS-1
; RUN: llvm-dis %t/m2.bc -o - | FileCheck %s --check-prefix=GUIDS-2
;
; GUIDS-1: name: "m1_f1"
; GUIDS-1-SAME: guid = 6019442868614718803
; GUIDS-2: name: "m2_f1"
; GUIDS-2-SAME: guid = 15593096274670919754
;
; RUN: rm -rf %t_baseline
; RUN: rm -rf %t_exp
; RUN: mkdir -p %t_baseline
; RUN: mkdir -p %t_exp
;
; Normal run. m1 shouldn't get m2_f1 because it's not referenced from there, and
; m1_f1 shouldn't go to m2.
;
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc \
; RUN:  -o %t_baseline/result.o -save-temps \
; RUN:  -r %t/m1.bc,m1_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1,plx
; RUN: llvm-dis %t_baseline/result.o.1.3.import.bc -o - | FileCheck %s --check-prefix=NOPROF-1
; RUN: llvm-dis %t_baseline/result.o.2.3.import.bc -o - | FileCheck %s --check-prefix=NOPROF-2
;
; NOPROF-1-NOT: m2_f1()
; NOPROF-2-NOT: m1_f1()
;
; The run with workload definitions - same other options.
;
; RUN: echo '[ \
; RUN:        {"Guid": 6019442868614718803, "Counters": [1], "Callsites": [[{"Guid": 15593096274670919754, "Counters": [1]}]]}, \
; RUN:        {"Guid": 15593096274670919754, "Counters": [1], "Callsites": [[{"Guid": 6019442868614718803, "Counters": [1]}]]} \
; RUN:  ]' > %t_exp/ctxprof.json
; RUN: llvm-ctxprof-util fromJSON --input %t_exp/ctxprof.json --output %t_exp/ctxprof.bitstream
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc \
; RUN:  -o %t_exp/result.o -save-temps \
; RUN:  -use-ctx-profile=%t_exp/ctxprof.bitstream \
; RUN:  -r %t/m1.bc,m1_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1,plx
; RUN: llvm-dis %t_exp/result.o.1.3.import.bc -o - | FileCheck %s --check-prefix=FIRST
; RUN: llvm-dis %t_exp/result.o.2.3.import.bc -o - | FileCheck %s --check-prefix=SECOND
;
;
; FIRST: m2_f1()
; SECOND: m1_f1()
;
;--- m1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local void @m1_f1() {
  ret void
}

;--- m2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local void @m2_f1() {
  ret void
}
