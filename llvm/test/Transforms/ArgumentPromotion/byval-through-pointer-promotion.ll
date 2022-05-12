; RUN: opt -passes=argpromotion -S %s | FileCheck %s

%struct.A = type { float, [12 x i8], i64, [8 x i8] }

define internal float @callee(%struct.A* byval(%struct.A) align 32 %0) {
; CHECK-LABEL: define {{[^@]+}}@callee
; CHECK-SAME: (float [[ARG_0:%.*]], i64 [[ARG_1:%.*]]) {
; CHECK-NEXT:    [[SUM:%.*]] = fadd float 0.000000e+00, [[ARG_0]]
; CHECK-NEXT:    [[COEFF:%.*]] = uitofp i64 [[ARG_1]] to float
; CHECK-NEXT:    [[RES:%.*]] = fmul float [[SUM]], [[COEFF]]
; CHECK-NEXT:    ret float [[RES]]
;
  %2 = getelementptr inbounds %struct.A, %struct.A* %0, i32 0, i32 0
  %3 = load float, float* %2, align 32
  %4 = fadd float 0.000000e+00, %3
  %5 = getelementptr inbounds %struct.A, %struct.A* %0, i32 0, i32 2
  %6 = load i64, i64* %5, align 16
  %7 = uitofp i64 %6 to float
  %8 = fmul float %4, %7
  ret float %8
}

define float @caller(float %0) {
; CHECK-LABEL: define {{[^@]+}}@caller
; CHECK-SAME: (float [[ARG_0:%.*]]) {
; CHECK-NEXT:    [[TMP_0:%.*]] = alloca %struct.A, align 32
; CHECK-NEXT:    [[FL_PTR_0:%.*]] = getelementptr inbounds %struct.A, %struct.A* [[TMP_0]], i32 0, i32 0
; CHECK-NEXT:    store float [[ARG_0]], float* [[FL_PTR_0]], align 32
; CHECK-NEXT:    [[I64_PTR_0:%.*]] = getelementptr inbounds %struct.A, %struct.A* [[TMP_0]], i32 0, i32 2
; CHECK-NEXT:    store i64 2, i64* [[I64_PTR_0]], align 16
; CHECK-NEXT:    [[FL_PTR_1:%.*]] = getelementptr %struct.A, %struct.A* [[TMP_0]], i64 0, i32 0
; CHECK-NEXT:    [[FL_VAL:%.*]] = load float, float* [[FL_PTR_1]], align 32
; CHECK-NEXT:    [[I64_PTR_1:%.*]] = getelementptr %struct.A, %struct.A* [[TMP_0]], i64 0, i32 2
; CHECK-NEXT:    [[I64_VAL:%.*]] = load i64, i64* [[I64_PTR_1]], align 16
; CHECK-NEXT:    [[RES:%.*]] = call noundef float @callee(float [[FL_VAL]], i64 [[I64_VAL]])
; CHECK-NEXT:    ret float [[RES]]
;
  %2 = alloca %struct.A, align 32
  %3 = getelementptr inbounds %struct.A, %struct.A* %2, i32 0, i32 0
  store float %0, float* %3, align 32
  %4 = getelementptr inbounds %struct.A, %struct.A* %2, i32 0, i32 2
  store i64 2, i64* %4, align 16
  %5 = call noundef float @callee(%struct.A* byval(%struct.A) align 32 %2)
  ret float %5
}
