; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -S %s | FileCheck %s --check-prefix=DENSE
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -pgo-instrument-dense-wave-counts=false -S %s | FileCheck %s --check-prefix=SPARSE
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -mtriple=x86_64-unknown-linux-gnu -S %s | FileCheck %s --check-prefix=SPARSE
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry=false -S %s | FileCheck %s --check-prefix=NOENTRY
; RUN: opt -passes=pgo-instr-gen -pgo-function-entry-coverage -S %s | FileCheck %s --check-prefix=ENTRY-COV --implicit-check-not='i64 0)'
; RUN: opt -passes=pgo-instr-gen -pgo-block-coverage -S %s | FileCheck %s --check-prefix=BLOCK-COV --implicit-check-not='i64 0)'
; RUN: opt -passes=pgo-instr-gen -pgo-temporal-instrumentation -S %s | FileCheck %s --check-prefix=TEMPORAL --implicit-check-not='i64 0)'
; RUN: not opt -passes='default<O2>' -cs-profilegen-file=dense-wave -cspgo-kind=cspgo-instr-gen-pipeline -pgo-instrument-entry -print-before=instrprof -disable-output %s 2>&1 | FileCheck %s --check-prefix=CS --implicit-check-not='i64 0)'

target triple = "amdgcn-amd-amdhsa"

; Dense generation keeps the original block/select prefix, then appends the
; unmeasured blocks in function order. The zero steps measure waves only.
; DENSE: @__llvm_profile_raw_version = {{.*}}constant i64 378302368699121676
; SPARSE: @__llvm_profile_raw_version = {{.*}}constant i64 360287970189639692
; NOENTRY: @__llvm_profile_raw_version = {{.*}}constant i64 90071992547409932
; ENTRY-COV: @__llvm_profile_raw_version = {{.*}}constant i64 3530822107858468876
; BLOCK-COV: @__llvm_profile_raw_version = {{.*}}constant i64 1224979098644774924
; TEMPORAL: @__llvm_profile_raw_version = {{.*}}constant i64 -9151314442816847860
; CS: @__llvm_profile_raw_version = {{.*}}constant i64 504403158265495564
; CS: LLVM ERROR: wave counts require ordinary counter increments
;
; DENSE-LABEL: define void @diamond
; DENSE: entry:
; DENSE-NEXT: call void @llvm.instrprof.increment({{.*}}i64 [[HASH:942389667449461396]], i32 5, i32 0)
; DENSE: a:
; DENSE-NEXT: call void @llvm.instrprof.increment.step({{.*}}i64 [[HASH]], i32 5, i32 3, i64 0)
; DENSE: b:
; DENSE-NEXT: call void @llvm.instrprof.increment({{.*}}i64 [[HASH]], i32 5, i32 1)
; DENSE: exit:
; DENSE-NEXT: call void @llvm.instrprof.increment.step({{.*}}i64 [[HASH]], i32 5, i32 4, i64 0)
; DENSE: call void @llvm.instrprof.increment.step({{.*}}i64 [[HASH]], i32 5, i32 2, i64 {{%.*}})
;
; SPARSE-LABEL: define void @diamond
; SPARSE: call void @llvm.instrprof.increment({{.*}}i32 3, i32 0)
; SPARSE: a:
; SPARSE-NEXT: store volatile
; SPARSE: call void @llvm.instrprof.increment({{.*}}i32 3, i32 1)
; SPARSE: exit:
; SPARSE: call void @llvm.instrprof.increment.step({{.*}}i32 3, i32 2, i64 {{%.*}})
;
; Even without forced entry instrumentation, the entry gets a direct wave site.
; NOENTRY-LABEL: define void @diamond
; NOENTRY: entry:
; NOENTRY-NEXT: call void @llvm.instrprof.increment.step({{.*}}i64 942389667449461396, i32 5, i32 3, i64 0)
define void @diamond(i1 %cond, i1 %select_cond, ptr %p) {
entry:
  br i1 %cond, label %a, label %b
a:
  store volatile i32 1, ptr %p
  br label %exit
b:
  store volatile i32 2, ptr %p
  br label %exit
exit:
  %value = select i1 %select_cond, i32 1, i32 2
  store volatile i32 %value, ptr %p
  ret void
}
