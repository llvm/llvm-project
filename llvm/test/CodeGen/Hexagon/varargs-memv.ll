; RUN: llc -march=hexagon < %s
; REQUIRES: asserts
; Check that llc does not crash.

@g0 = private unnamed_addr constant [7 x i8] c"%d\09\09%d\00", align 1
@g1 = common global <4 x i32> zeroinitializer, align 16

declare i32 @f0(...)

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca [0 x <4 x i32>], align 16
  store i32 0, i32* %v0
  store i32 0, i32* %v1, align 4
  %v3 = bitcast [0 x <4 x i32>]* %v2 to i8*
  call void @llvm.memset.p0i8.i32(i8* align 16 %v3, i8 0, i32 0, i1 false)
  %v4 = load i32, i32* %v1, align 4
  %v5 = add nsw i32 %v4, 1
  store i32 %v5, i32* %v1, align 4
  %v6 = load <4 x i32>, <4 x i32>* @g1, align 16
  %v7 = call i32 bitcast (i32 (...)* @f0 to i32 (i8*, i32, <4 x i32>)*)(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @g0, i32 0, i32 0), i32 %v5, <4 x i32> %v6)
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { argmemonly nounwind }
