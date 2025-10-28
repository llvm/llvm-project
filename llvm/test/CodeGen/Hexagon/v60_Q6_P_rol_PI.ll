; RUN: llc -mtriple=hexagon -O0 < %s | FileCheck %s
; CHECK: r{{[0-9]*}}:{{[0-9]*}} = rol(r{{[0-9]*}}:{{[0-9]*}},#4)

target triple = "hexagon"

@g0 = private unnamed_addr constant [33 x i8] c"%llx :  Q6_P_rol_PI(LONG_MIN,0)\0A\00", align 1

; Function Attrs: nounwind
declare i32 @f0(ptr, ...) #0

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  store i32 0, ptr %v0
  store i32 0, ptr %v1, align 4
  %v2 = call i64 @llvm.hexagon.S6.rol.i.p(i64 483648, i32 4)
  %v3 = call i32 (ptr, ...) @f0(ptr @g0, i64 %v2) #2
  ret i32 0
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S6.rol.i.p(i64, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
