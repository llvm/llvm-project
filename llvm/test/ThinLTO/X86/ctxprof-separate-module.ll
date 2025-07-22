; REQUIRES: asserts
; Test workload based importing via -thinlto-pgo-ctx-prof with moving the whole
; graph to a new module.
; Use external linkage symbols so we don't depend on module paths which are
; used when computing the GUIDs of internal linkage symbols.
;
; Set up
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: split-file %s %t
;
; RUN: opt -module-summary -passes=assign-guid,ctx-instr-gen %t/m1.ll -o %t/m1.bc
; RUN: opt -module-summary -passes=assign-guid,ctx-instr-gen %t/m2.ll -o %t/m2.bc
; RUN: opt -module-summary -passes=assign-guid,ctx-instr-gen %t/m3.ll -o %t/m3.bc
; RUN: opt -module-summary -passes=assign-guid,ctx-instr-gen %t/6019442868614718803.ll -o %t/6019442868614718803.bc

; RUN: llvm-ctxprof-util fromYAML --input %t/ctxprof.yaml --output %t/ctxprof.bitstream
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc %t/m3.bc %t/6019442868614718803.bc -thinlto-move-ctxprof-trees \
; RUN:  -o %t/result.o -save-temps \
; RUN:  -use-ctx-profile=%t/ctxprof.bitstream \
; RUN:  -r %t/m1.bc,m1_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1,plx \
; RUN:  -r %t/m3.bc,m1_f1 \
; RUN:  -r %t/m3.bc,m3_f1,plx -debug-only=function-import 2>&1 | FileCheck %s --check-prefix=ABSENT-MSG

; also add the move semantics for the root:
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc %t/m3.bc %t/6019442868614718803.bc -thinlto-move-ctxprof-trees \
; RUN:  -thinlto-move-symbols=6019442868614718803 \
; RUN:  -o %t/result-with-move.o -save-temps \
; RUN:  -use-ctx-profile=%t/ctxprof.bitstream \
; RUN:  -r %t/m1.bc,m1_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1,plx \
; RUN:  -r %t/m3.bc,m1_f1 \
; RUN:  -r %t/m3.bc,m3_f1,plx -debug-only=function-import 2>&1 | FileCheck %s --check-prefix=ABSENT-MSG

; RUN: llvm-dis %t/result.o.4.3.import.bc -o - | FileCheck %s
; RUN: llvm-dis %t/result.o.3.3.import.bc -o - | FileCheck %s --check-prefix=ABSENT
; RUN: llvm-dis %t/result-with-move.o.1.3.import.bc -o - | FileCheck %s --check-prefix=WITHMOVE-SRC
; RUN: llvm-dis %t/result-with-move.o.4.3.import.bc -o - | FileCheck %s --check-prefix=WITHMOVE-DEST
; RUN: llvm-dis %t/result.o.1.3.import.bc -o - | FileCheck %s --check-prefix=WITHOUTMOVE-SRC
;
; CHECK: define available_externally void @m1_f1()
; CHECK: define available_externally void @m2_f1()
; ABSENT: declare void @m1_f1()
; ABSENT-MSG: Skipping over 6019442868614718803 because its import is handled in a different module.
;
; WITHMOVE-SRC: declare dso_local void @m1_f1
; WITHMOVE-DEST: define dso_local void @m1_f1
; WITHOUTMOVE-SRC: define dso_local void @m1_f1
;--- ctxprof.yaml
Contexts: 
  -
    Guid: 6019442868614718803
    TotalRootEntryCount: 5
    Counters: [1]
    Callsites:  
      - -
          Guid: 15593096274670919754
          Counters: [1]

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

;--- m3.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @m1_f1()

define dso_local void @m3_f1() {
  call void @m1_f1()
  ret void
}

;--- 6019442868614718803.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
