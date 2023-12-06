; RUN: mkdir -p %t_baseline
; RUN: echo '{"m1_f1":["m2_f1", "m2_f1_alias", "interposable_f", "noninterposable_f"], \
; RUN:  "m2_f1":["m1_f1", "m1_f2"]}' > %t/workload_defs.json
; RUN: opt -module-summary %S/Inputs/workload1.ll -o %t_baseline/workload1.bc
; RUN: opt -module-summary %S/Inputs/workload2.ll -o %t_baseline/workload2.bc
; RUN: opt -module-summary %S/Inputs/workload3.ll -o %t_baseline/workload3.bc
;
; Normal run. The first module shouldn't get m2_f1
;
; RUN: llvm-lto2 run %t_baseline/workload1.bc %t_baseline/workload2.bc %t_baseline/workload3.bc \
; RUN:  -o %t_baseline/result.o -save-temps \
; RUN:  -r %t_baseline/workload1.bc,m1_f1,plx \
; RUN:  -r %t_baseline/workload1.bc,interposable_f \
; RUN:  -r %t_baseline/workload1.bc,noninterposable_f \
; RUN:  -r %t_baseline/workload1.bc,m1_variant \
; RUN:  -r %t_baseline/workload2.bc,m2_f1,plx \
; RUN:  -r %t_baseline/workload2.bc,m2_f1_alias \
; RUN:  -r %t_baseline/workload2.bc,interposable_f,p \
; RUN:  -r %t_baseline/workload2.bc,noninterposable_f,p \
; RUN:  -r %t_baseline/workload2.bc,m2_variant \
; RUN:  -r %t_baseline/workload3.bc,m1_f1 \
; RUN:  -r %t_baseline/workload3.bc,m3_f1,plx
; RUN: llvm-dis %t_baseline/result.o.1.3.import.bc -o - | FileCheck %s --check-prefix=NOPROF
;
; NOPROF-NOT: m2_f1
;
; The run with workload definitions - same other options.
;
; RUN: mkdir -p %t
; RUN: llvm-lto2 run %t/workload1.bc %t/workload2.bc %t/workload3.bc \
; RUN:  -thinlto-workload-def=%t/workload_defs.json -o %t/result.o -save-temps \
; RUN:  -r %t/workload1.bc,m1_f1,plx \
; RUN:  -r %t/workload1.bc,interposable_f \
; RUN:  -r %t/workload1.bc,noninterposable_f \
; RUN:  -r %t/workload1.bc,m1_variant \
; RUN:  -r %t/workload2.bc,m2_f1,plx \
; RUN:  -r %t/workload2.bc,m2_f1_alias \
; RUN:  -r %t/workload2.bc,interposable_f,p \
; RUN:  -r %t/workload2.bc,noninterposable_f,p \
; RUN:  -r %t/workload2.bc,m2_variant \
; RUN:  -r %t/workload3.bc,m1_f1 \
; RUN:  -r %t/workload3.bc,m3_f1,plx
; RUN: llvm-dis %t/result.o.1.3.import.bc -o - | FileCheck %s --check-prefix=FIRST
; RUN: llvm-dis %t/result.o.2.3.import.bc -o - | FileCheck %s --check-prefix=SECOND
; RUN: llvm-dis %t/result.o.3.3.import.bc -o - | FileCheck %s --check-prefix=THIRD
;
; The third module is bitwse-identical to the "normal" run, as the 
; RUN: diff %t_baseline/result.o.3.3.import.bc %t/result.o.3.3.import.bc
;
; This time, we expect m1 to have m2_f1 and the m2 variant of interposable_f,
; while keeping its variant of noninterposable_f
;
; FIRST-LABEL: @m1_f1
; FIRST-LABEL: @m1_f2
; FIRST-LABEL: define available_externally void @noninterposable_f
; FIRST-NEXT: call void @m1_variant
; FIRST-LABEL: @m2_f1
; FIRST-LABEL: define available_externally void @interposable_f
; FIRST-NEXT: call void @m2_variant
; FIRST-NOT:   @m2_f1_alias
; SECOND-LABEL: @m2_f1
; SECOND-LABEL: @m1_f1
; SECOND-LABEL: @m1_f2
; THIRD-LABEL: define available_externally void @m1_f1