; Set up
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: split-file %s %t
;
;--- m1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @m1_variant()
declare void @m2_f1_alias()

define dso_local void @m1_f1() {
  call void @m1_f2()
  call void @noninterposable_f()
  ret void
}

define internal void @m1_f2() {
  call void @interposable_f()
  ret void
}

define linkonce void @interposable_f() {
  call void @m1_variant()
  ret void
}

define linkonce_odr void @noninterposable_f() {
  call void @m1_variant()
  ret void
}
;--- m2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @m2_variant()

define dso_local void @m2_f1() {
  call void @interposable_f()
  call void @noninterposable_f()
  ret void
}

@m2_f1_alias = alias void (...), ptr @m2_f1

define external void @interposable_f() {
  call void @m2_variant() 
  ret void
}

define linkonce_odr void @noninterposable_f() {
  call void @m2_variant()
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
;
; RUN: opt -module-summary %t/m1.ll -o %t/m1.bc
; RUN: opt -module-summary %t/m2.ll -o %t/m2.bc
; RUN: opt -module-summary %t/m3.ll -o %t/m3.bc
; RUN: rm -rf %t_baseline
; RUN: rm -rf %t_exp
; RUN: mkdir -p %t_baseline
; RUN: mkdir -p %t_exp
;
; Normal run. m1 shouldn't get m2_f1 because it's not referenced from there.
;
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc %t/m3.bc \
; RUN:  -o %t_baseline/result.o -save-temps \
; RUN:  -r %t/m1.bc,m1_f1,plx \
; RUN:  -r %t/m1.bc,interposable_f \
; RUN:  -r %t/m1.bc,noninterposable_f \
; RUN:  -r %t/m1.bc,m1_variant \
; RUN:  -r %t/m1.bc,m2_f1_alias \
; RUN:  -r %t/m2.bc,m2_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1_alias,plx \
; RUN:  -r %t/m2.bc,interposable_f,p \
; RUN:  -r %t/m2.bc,noninterposable_f,p \
; RUN:  -r %t/m2.bc,m2_variant \
; RUN:  -r %t/m3.bc,m1_f1 \
; RUN:  -r %t/m3.bc,m3_f1,plx
; RUN: llvm-dis %t_baseline/result.o.1.3.import.bc -o - | FileCheck %s --check-prefix=NOPROF
;
; NOPROF-NOT: m2_f1()
;
; The run with workload definitions - same other options.
;
; RUN: echo '{"m1_f1":["m2_f1", "m2_f1_alias", "interposable_f", "noninterposable_f"], \
; RUN:  "m2_f1":["m1_f1", "m1_f2"]}' > %t_exp/workload_defs.json
;
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc %t/m3.bc \
; RUN:  -o %t_exp/result.o -save-temps \
; RUN:  -thinlto-workload-def=%t_exp/workload_defs.json \
; RUN:  -r %t/m1.bc,m1_f1,plx \
; RUN:  -r %t/m1.bc,interposable_f \
; RUN:  -r %t/m1.bc,noninterposable_f \
; RUN:  -r %t/m1.bc,m1_variant \
; RUN:  -r %t/m1.bc,m2_f1_alias \
; RUN:  -r %t/m2.bc,m2_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1_alias,plx \
; RUN:  -r %t/m2.bc,interposable_f,p \
; RUN:  -r %t/m2.bc,noninterposable_f,p \
; RUN:  -r %t/m2.bc,m2_variant \
; RUN:  -r %t/m3.bc,m1_f1 \
; RUN:  -r %t/m3.bc,m3_f1,plx
; RUN: llvm-dis %t_exp/result.o.1.3.import.bc -o - | FileCheck %s --check-prefix=FIRST
; RUN: llvm-dis %t_exp/result.o.2.3.import.bc -o - | FileCheck %s --check-prefix=SECOND
; RUN: llvm-dis %t_exp/result.o.3.3.import.bc -o - | FileCheck %s --check-prefix=THIRD
;
; The third module is bitwse-identical to the "normal" run, as the workload
; defintion doesn't mention it.
;
; RUN: diff %t_baseline/result.o.3.3.import.bc %t_exp/result.o.3.3.import.bc
;
; This time, we expect m1 to have m2_f1 and the m2 variant of interposable_f,
; while keeping its variant of noninterposable_f
;
; FIRST-LABEL:  @m1_f1
; FIRST-LABEL:  @m1_f2
; FIRST-LABEL:  define available_externally void @noninterposable_f
; FIRST-NEXT:   call void @m1_variant
; FIRST-LABEL:  @m2_f1
; FIRST-LABEL:  define available_externally void @interposable_f
; FIRST-NEXT:   call void @m2_variant
; FIRST-LABEL:  @m2_f1_alias
; SECOND-LABEL: @m2_f1
; SECOND-LABEL: @m1_f1
; SECOND-LABEL: @m1_f2
; THIRD-LABEL: define available_externally void @m1_f1