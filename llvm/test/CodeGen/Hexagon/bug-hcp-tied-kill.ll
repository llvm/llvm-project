; RUN: llc -mtriple=hexagon -O2 -verify-machineinstrs < %s | FileCheck %s

; CHECK: .globl

target triple = "hexagon"

@g0 = private unnamed_addr constant [46 x i8] c"%x :  Q6_R_mpyiacc_RR(INT32_MAX,0,INT32_MAX)\0A\00", align 1
@g1 = private unnamed_addr constant [46 x i8] c"%x :  Q6_R_mpyiacc_RR(INT32_MIN,1,INT32_MAX)\0A\00", align 1
@g2 = private unnamed_addr constant [39 x i8] c"%x :  Q6_R_mpyiacc_RR(-1,1,INT32_MAX)\0A\00", align 1
@g3 = private unnamed_addr constant [38 x i8] c"%x :  Q6_R_mpyiacc_RR(0,1,INT32_MAX)\0A\00", align 1
@g4 = private unnamed_addr constant [38 x i8] c"%x :  Q6_R_mpyiacc_RR(1,1,INT32_MAX)\0A\00", align 1
@g5 = private unnamed_addr constant [46 x i8] c"%x :  Q6_R_mpyiacc_RR(INT32_MAX,1,INT32_MAX)\0A\00", align 1
@g6 = private unnamed_addr constant [54 x i8] c"%x :  Q6_R_mpyiacc_RR(INT32_MIN,INT32_MAX,INT32_MAX)\0A\00", align 1
@g7 = private unnamed_addr constant [47 x i8] c"%x :  Q6_R_mpyiacc_RR(-1,INT32_MAX,INT32_MAX)\0A\00", align 1
@g8 = private unnamed_addr constant [46 x i8] c"%x :  Q6_R_mpyiacc_RR(0,INT32_MAX,INT32_MAX)\0A\00", align 1
@g9 = private unnamed_addr constant [46 x i8] c"%x :  Q6_R_mpyiacc_RR(1,INT32_MAX,INT32_MAX)\0A\00", align 1
@g10 = private unnamed_addr constant [54 x i8] c"%x :  Q6_R_mpyiacc_RR(INT32_MAX,INT32_MAX,INT32_MAX)\0A\00", align 1

; Function Attrs: nounwind
declare i32 @f0(ptr nocapture readonly, ...) #0

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  %v0 = tail call i32 @llvm.hexagon.M2.maci(i32 2147483647, i32 0, i32 2147483647)
  %v1 = tail call i32 (ptr, ...) @f0(ptr @g0, i32 %v0) #2
  %v2 = tail call i32 @llvm.hexagon.M2.maci(i32 -2147483648, i32 1, i32 2147483647)
  %v3 = tail call i32 (ptr, ...) @f0(ptr @g1, i32 %v2) #2
  %v4 = tail call i32 @llvm.hexagon.M2.maci(i32 -1, i32 1, i32 2147483647)
  %v5 = tail call i32 (ptr, ...) @f0(ptr @g2, i32 %v4) #2
  %v6 = tail call i32 @llvm.hexagon.M2.maci(i32 0, i32 1, i32 2147483647)
  %v7 = tail call i32 (ptr, ...) @f0(ptr @g3, i32 %v6) #2
  %v8 = tail call i32 @llvm.hexagon.M2.maci(i32 1, i32 1, i32 2147483647)
  %v9 = tail call i32 (ptr, ...) @f0(ptr @g4, i32 %v8) #2
  %v10 = tail call i32 @llvm.hexagon.M2.maci(i32 2147483647, i32 1, i32 2147483647)
  %v11 = tail call i32 (ptr, ...) @f0(ptr @g5, i32 %v10) #2
  %v12 = tail call i32 @llvm.hexagon.M2.maci(i32 -2147483648, i32 2147483647, i32 2147483647)
  %v13 = tail call i32 (ptr, ...) @f0(ptr @g6, i32 %v12) #2
  %v14 = tail call i32 @llvm.hexagon.M2.maci(i32 -1, i32 2147483647, i32 2147483647)
  %v15 = tail call i32 (ptr, ...) @f0(ptr @g7, i32 %v14) #2
  %v16 = tail call i32 @llvm.hexagon.M2.maci(i32 0, i32 2147483647, i32 2147483647)
  %v17 = tail call i32 (ptr, ...) @f0(ptr @g8, i32 %v16) #2
  %v18 = tail call i32 @llvm.hexagon.M2.maci(i32 1, i32 2147483647, i32 2147483647)
  %v19 = tail call i32 (ptr, ...) @f0(ptr @g9, i32 %v18) #2
  %v20 = tail call i32 @llvm.hexagon.M2.maci(i32 2147483647, i32 2147483647, i32 2147483647)
  %v21 = tail call i32 (ptr, ...) @f0(ptr @g10, i32 %v20) #2
  ret i32 0
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.maci(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
