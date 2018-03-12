; RUN: llc -O0 -march=hexagon < %s | FileCheck %s

; CHECK: [[REG0:(r[0-9]+)]] = add(r{{[0-9]+}},mpyi([[REG0]],r{{[0-9]+}})
; CHECK: [[REG0:(r[0-9]+)]] = add(r{{[0-9]+}},mpyi([[REG0]],r{{[0-9]+}})

target triple = "hexagon"

@g0 = private unnamed_addr constant [50 x i8] c"%x :  Q6_R_add_mpyi_RRR(INT_MIN,INT_MIN,INT_MIN)\0A\00", align 1
@g1 = private unnamed_addr constant [45 x i8] c"%x :  Q6_R_add_mpyi_RRR(-1,INT_MIN,INT_MIN)\0A\00", align 1

; Function Attrs: nounwind
declare i32 @f0(i8* nocapture readonly, ...) #0

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  %v0 = tail call i32 @llvm.hexagon.M4.mpyrr.addr(i32 -2147483648, i32 -2147483648, i32 -2147483648)
  %v1 = tail call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([50 x i8], [50 x i8]* @g0, i32 0, i32 0), i32 %v0) #2
  %v2 = tail call i32 @llvm.hexagon.M4.mpyrr.addr(i32 -1, i32 -2147483648, i32 -2147483648)
  %v3 = tail call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @g1, i32 0, i32 0), i32 %v2) #2
  ret i32 0
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M4.mpyrr.addr(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
