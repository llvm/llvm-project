; Check that the __stack_chk_guard was placed in small data.
; RUN: llc -march=hexagon -mtriple=hexagon-unknown-linux-gnu -O2 -hexagon-small-data-threshold=4 < %s | FileCheck -check-prefix=GPREL %s
; GPREL: memw(gp+#__stack_chk_guard)

; For threshold less than 4 (size of address), the variable is not placed in small-data
; RUN: llc -march=hexagon -mtriple=hexagon-unknown-linux-gnu -O2 -hexagon-small-data-threshold=0 < %s | FileCheck -check-prefix=ABS %s
; ABS: memw(##__stack_chk_guard)

@g0 = private unnamed_addr constant [37 x i8] c"This string is longer than 16 bytes\0A\00", align 1
@g1 = private unnamed_addr constant [15 x i8] c"\0AChar 20 = %c\0A\00", align 1

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
  %v5 = call i8* @f1(i8* %v3, i8* %v4) #2
  %v6 = load i32, i32* %v0, align 4
  %v7 = getelementptr inbounds [64 x i8], [64 x i8]* %v1, i32 0, i32 %v6
  %v8 = load i8, i8* %v7, align 1
  ret i8 %v8
}

; Function Attrs: nounwind
declare i8* @f1(i8*, i8*) #1

; Function Attrs: noinline nounwind ssp
define i32 @f2(i32 %a0, i8** %a1) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i8**, align 4
  store i32 0, i32* %v0, align 4
  store i32 %a0, i32* %v1, align 4
  store i8** %a1, i8*** %v2, align 4
  %v3 = call zeroext i8 @f0(i32 20)
  %v4 = zext i8 %v3 to i32
  %v5 = call i32 (i8*, ...) @f3(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @g1, i32 0, i32 0), i32 %v4) #2
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @f3(i8*, ...) #1

attributes #0 = { noinline nounwind ssp "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #2 = { nounwind }
