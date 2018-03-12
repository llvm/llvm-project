; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]*}} += sfmpy(r{{[0-9]*}},r{{[0-9]*}},p{{[0-3]}}):scale

target triple = "hexagon"

@g0 = private unnamed_addr constant [65 x i8] c"%f :  Q6_R_sfmpyacc_RRp_scale(FLT_MIN,FLT_MIN,FLT_MIN,CHAR_MIN)\0A\00", align 1

; Function Attrs: nounwind
declare i32 @f0(i8*, ...) #0

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  store i32 0, i32* %v0
  store i32 0, i32* %v1, align 4
  %v2 = call float @llvm.hexagon.F2.sffma.sc(float 0x3810000000000000, float 0x3810000000000000, float 0x3810000000000000, i32 0)
  %v3 = fpext float %v2 to double
  %v4 = call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @g0, i32 0, i32 0), double %v3) #0
  ret i32 0
}

; Function Attrs: readnone
declare float @llvm.hexagon.F2.sffma.sc(float, float, float, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { readnone }
