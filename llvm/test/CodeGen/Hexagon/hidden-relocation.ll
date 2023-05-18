; RUN: llc -march=hexagon -O2 -relocation-model=pic < %s | FileCheck %s
;
; CHECK: r{{[0-9]+}} = add({{pc|PC}},##g2@PCREL)

@g0 = hidden global i32 10, align 4
@g1 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@g2 = internal global ptr @g0, align 4

; Function Attrs: nounwind
declare i32 @f0(ptr, ...) #0

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 10, ptr @g0, align 4
  %v1 = load ptr, ptr @g2, align 4
  %v2 = load i32, ptr %v1, align 4
  %v3 = call i32 (ptr, ...) @f0(ptr @g1, i32 %v2)
  %v4 = load i32, ptr %v0
  ret i32 %v4
}

attributes #0 = { nounwind }
