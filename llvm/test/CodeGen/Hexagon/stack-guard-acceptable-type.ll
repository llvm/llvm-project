; Check that we accept a user definition/declaration of __stack_chk_guard
; that is not the expected type (i8*) but one of the same size.
;
; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK: __stack_chk_fail

target triple = "hexagon"

@__stack_chk_guard = external global i32, align 4
@g0 = private unnamed_addr constant [37 x i8] c"This string is longer than 16 bytes\0A\00", align 1

; Function Attrs: noinline nounwind ssp
define zeroext i8 @f0(i32 %a0) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca [64 x i8], align 8
  %v2 = alloca i8*, align 4
  store i32 %a0, i32* %v0, align 4
  store i8* getelementptr inbounds ([37 x i8], [37 x i8]* @g0, i32 0, i32 0), i8** %v2, align 4
  %v3 = getelementptr inbounds [64 x i8], [64 x i8]* %v1, i32 0, i32 0
  %v4 = load i8*, i8** %v2, align 4
  %v5 = call i8* @f1(i8* %v3, i8* %v4) #1
  %v6 = load i32, i32* %v0, align 4
  %v7 = getelementptr inbounds [64 x i8], [64 x i8]* %v1, i32 0, i32 %v6
  %v8 = load i8, i8* %v7, align 1
  ret i8 %v8
}

; Function Attrs: nounwind
declare i8* @f1(i8*, i8*) #1

attributes #0 = { noinline nounwind ssp }
attributes #1 = { nounwind }
