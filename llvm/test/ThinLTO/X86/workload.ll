; Test workload based importing via -thinlto-workload-def
;
; Set up
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: split-file %s %t
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
; RUN:  -r %t/m1.bc,interposable_f,p \
; RUN:  -r %t/m1.bc,noninterposable_f \
; RUN:  -r %t/m1.bc,m1_variant \
; RUN:  -r %t/m1.bc,m2_f1_alias \
; RUN:  -r %t/m2.bc,m2_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1_alias,plx \
; RUN:  -r %t/m2.bc,interposable_f \
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
; RUN: echo '{ \
; RUN:    "m1_f1": ["m1_f1", "m2_f1", "m2_f1_alias", "interposable_f", "noninterposable_f"], \
; RUN:    "m2_f1": ["m1_f1", "m1_f2", "interposable_f"] \
; RUN:  }' > %t_exp/workload_defs.json
;
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc %t/m3.bc \
; RUN:  -o %t_exp/result.o -save-temps \
; RUN:  -thinlto-workload-def=%t_exp/workload_defs.json \
; RUN:  -r %t/m1.bc,m1_f1,plx \
; RUN:  -r %t/m1.bc,interposable_f,p \
; RUN:  -r %t/m1.bc,noninterposable_f \
; RUN:  -r %t/m1.bc,m1_variant \
; RUN:  -r %t/m1.bc,m2_f1_alias \
; RUN:  -r %t/m2.bc,m2_f1,plx \
; RUN:  -r %t/m2.bc,m2_f1_alias,plx \
; RUN:  -r %t/m2.bc,interposable_f \
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
; This time, we expect m1 to have m2_f1 and the m2 variant of both interposable_f
; and noninterposable_f
;
; FIRST-LABEL:  @m1_f1
; FIRST-LABEL:  @m1_f2.llvm.0
;
; @interposable_f is prevailing in m1, so it won't be imported
; FIRST-LABEL:  define void @interposable_f
; FIRST-NEXT:   call void @m1_variant
;
; FIRST-LABEL:  @m2_f1
;
; @noninterposable_f is prevailing in m2 so it will be imported from there. 
; FIRST-LABEL:  define available_externally void @noninterposable_f
; FIRST-NEXT:   call void @m2_variant
;
; FIRST-LABEL:  define available_externally void @m2_f1_alias
;
; For the second module we expect to get the functions imported from m1: m1_f1
; and m1_f2. interposable_f will also come from m1 because that's where its
; prevailing variant is.
; SECOND-LABEL: @m2_f1
;
; SECOND-LABEL: define weak_odr void @noninterposable_f
; SECOND-NEXT:  call void @m2_variant()
; SECOND-LABEL: @m1_f1
; SECOND-LABEL: define available_externally hidden void @m1_f2.llvm.0
;
; we import @interposable_f from m1, the prevailing variant.
; SECOND-LABEL: define available_externally void @interposable_f
; SECOND-NEXT:  call void @m1_variant
;
; The third module remains unchanged. The more robust test is the `diff` test
; in the run lines above.
; THIRD-LABEL: define available_externally void @m1_f1

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

define external void @interposable_f() {
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

define weak void @interposable_f() {
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
