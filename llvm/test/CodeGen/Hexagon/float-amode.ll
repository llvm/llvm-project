; RUN: llc -march=hexagon -fp-contract=fast -disable-hexagon-peephole -disable-hexagon-amodeopt < %s | FileCheck %s

; The test checks for various addressing modes for floating point loads/stores.

%struct.matrix_paramsGlob = type { [50 x i8], i16, [50 x float] }
%struct.matrix_params = type { [50 x i8], i16, ptr }
%struct.matrix_params2 = type { i16, [50 x [50 x float]] }

@globB = common global %struct.matrix_paramsGlob zeroinitializer, align 4
@globA = common global %struct.matrix_paramsGlob zeroinitializer, align 4
@b = common global float 0.000000e+00, align 4
@a = common global float 0.000000e+00, align 4

; CHECK-LABEL: test1
; CHECK: [[REG11:(r[0-9]+)]] = memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2)
; CHECK: [[REG12:(r[0-9]+)]] += sfmpy({{.*}}[[REG11]]
; CHECK: memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2) = [[REG12]].new

; Function Attrs: norecurse nounwind
define void @test1(ptr nocapture readonly %params, i32 %col1) {
entry:
  %matrixA = getelementptr inbounds %struct.matrix_params, ptr %params, i32 0, i32 2
  %0 = load ptr, ptr %matrixA, align 4
  %arrayidx = getelementptr inbounds ptr, ptr %0, i32 2
  %1 = load ptr, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %1, i32 %col1
  %2 = load float, ptr %arrayidx1, align 4
  %mul = fmul float %2, 2.000000e+01
  %add = fadd float %mul, 1.000000e+01
  %arrayidx3 = getelementptr inbounds ptr, ptr %0, i32 5
  %3 = load ptr, ptr %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds float, ptr %3, i32 %col1
  store float %add, ptr %arrayidx4, align 4
  ret void
}

; CHECK-LABEL: test2
; CHECK: [[REG21:(r[0-9]+)]] = memw(##globB+92)
; CHECK: [[REG22:(r[0-9]+)]] = sfadd({{.*}}[[REG21]]
; CHECK: memw(##globA+84) = [[REG22]]

; Function Attrs: norecurse nounwind
define void @test2(ptr nocapture readonly %params, i32 %col1) {
entry:
  %matrixA = getelementptr inbounds %struct.matrix_params, ptr %params, i32 0, i32 2
  %0 = load ptr, ptr %matrixA, align 4
  %1 = load ptr, ptr %0, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %1, i32 %col1
  %2 = load float, ptr %arrayidx1, align 4
  %3 = load float, ptr getelementptr inbounds (%struct.matrix_paramsGlob, ptr @globB, i32 0, i32 2, i32 10), align 4
  %add = fadd float %2, %3
  store float %add, ptr getelementptr inbounds (%struct.matrix_paramsGlob, ptr @globA, i32 0, i32 2, i32 8), align 4
  ret void
}

; CHECK-LABEL: test3
; CHECK: [[REG31:(r[0-9]+)]] = memw(gp+#b)
; CHECK: [[REG32:(r[0-9]+)]] = sfadd({{.*}}[[REG31]]
; CHECK: memw(gp+#a) = [[REG32]]

; Function Attrs: norecurse nounwind
define void @test3(ptr nocapture readonly %params, i32 %col1) {
entry:
  %matrixA = getelementptr inbounds %struct.matrix_params, ptr %params, i32 0, i32 2
  %0 = load ptr, ptr %matrixA, align 4
  %1 = load ptr, ptr %0, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %1, i32 %col1
  %2 = load float, ptr %arrayidx1, align 4
  %3 = load float, ptr @b, align 4
  %add = fadd float %2, %3
  store float %add, ptr @a, align 4
  ret void
}

; CHECK-LABEL: test4
; CHECK: [[REG41:(r[0-9]+)]] = memw(r0<<#2+##globB+52)
; CHECK: [[REG42:(r[0-9]+)]] = sfadd({{.*}}[[REG41]]
; CHECK: memw(r0<<#2+##globA+60) = [[REG42]]
; Function Attrs: noinline norecurse nounwind
define void @test4(i32 %col1) {
entry:
  %arrayidx = getelementptr inbounds %struct.matrix_paramsGlob, ptr @globB, i32 0, i32 2, i32 %col1
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd float %0, 0.000000e+00
  %add1 = add nsw i32 %col1, 2
  %arrayidx2 = getelementptr inbounds %struct.matrix_paramsGlob, ptr @globA, i32 0, i32 2, i32 %add1
  store float %add, ptr %arrayidx2, align 4
  ret void
}
