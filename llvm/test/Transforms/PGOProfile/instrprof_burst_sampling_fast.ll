; RUN: opt < %s --passes=instrprof --sampled-instrumentation -S | FileCheck %s --check-prefixes=SAMPLE-VAR,SAMPLE-CODE,SAMPLE-DURATION,SAMPLE-WEIGHT
; RUN: opt < %s --passes=instrprof --sampled-instrumentation --sampled-instr-burst-duration=100 -S | FileCheck %s --check-prefixes=SAMPLE-VAR,SAMPLE-CODE,SAMPLE-DURATION100,SAMPLE-WEIGHT100

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__llvm_profile_raw_version = comdat any

; SAMPLE-VAR: $__llvm_profile_sampling = comdat any

@__llvm_profile_raw_version = constant i64 72057594037927940, comdat
@__profn_f = private constant [1 x i8] c"f"

; SAMPLE-VAR: @__llvm_profile_sampling = thread_local global i16 0, comdat
; SAMPLE-VAR: @__profc_f = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
; SAMPLE-VAR: @__profd_f = private global { i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i32 } { i64 -3706093650706652785, i64 12884901887, i64 sub (i64 ptrtoint (ptr @__profc_f to i64), i64 ptrtoint (ptr @__profd_f to i64)), i64 0, ptr @f.local, ptr null, i32 1, [3 x i16] zeroinitializer, i32 0 }, section "__llvm_prf_data", comdat($__profc_f), align 8
; SAMPLE-VAR: @__llvm_prf_nm = private constant {{.*}}, section "__llvm_prf_names", align 1
; SAMPLE-VAR: @llvm.compiler.used = appending global [2 x ptr] [ptr @__llvm_profile_sampling, ptr @__profd_f], section "llvm.metadata"
; SAMPLE-VAR: @llvm.used = appending global [1 x ptr] [ptr @__llvm_prf_nm], section "llvm.metadata"


define void @f() {
; SAMPLE-CODE-LABEL: @f(
; SAMPLE-CODE:  entry:
; SAMPLE-CODE-NEXT:    [[TMP0:%.*]] = load i16, ptr @__llvm_profile_sampling, align 2
; SAMPLE-DURATION:         [[TMP1:%.*]] = icmp ule i16 [[TMP0]], 200
; SAMPLE-DURATION100:     [[TMP1:%.*]] = icmp ule i16 [[TMP0]], 100
; SAMPLE-CODE:         br i1 [[TMP1]], label %[[TMP2:.*]], label %[[TMP4:.*]], !prof !0
; SAMPLE-CODE:       [[TMP2]]:
; SAMPLE-CODE-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_f
; SAMPLE-CODE-NEXT:    [[TMP3:%.*]] = add i64 [[PGOCOUNT]], 1
; SAMPLE-CODE-NEXT:    store i64 [[TMP3]], ptr @__profc_f
; SAMPLE-CODE-NEXT:    br label %[[TMP4]]
; SAMPLE-CODE:       [[TMP4]]:
; SAMPLE-CODE-NEXT:    [[TMP5:%.*]] = add i16 [[TMP0]], 1
; SAMPLE-CODE-NEXT:    store i16 [[TMP5]],  ptr @__llvm_profile_sampling, align 2
; SAMPLE-CODE-NEXT:    ret void
;
entry:
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__profn_f, i32 0, i32 0), i64 12884901887, i32 1, i32 0)
  ret void
}

; SAMPLE-WEIGHT: !0 = !{!"branch_weights", i32 200, i32 65336}
; SAMPLE-WEIGHT100: !0 = !{!"branch_weights", i32 100, i32 65436}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)
