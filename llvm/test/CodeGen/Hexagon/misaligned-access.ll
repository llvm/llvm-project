; RUN: llc -march=hexagon < %s
; Check that the mis-aligned load doesn't cause compiler to assert.

@g0 = common global i32 0, align 4

declare i32 @f0(i64) #0

define i32 @f1() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = load i32, i32* @g0, align 4
  store i32 %v1, i32* %v0, align 4
  %v2 = bitcast i32* %v0 to i64*
  %v3 = load i64, i64* %v2, align 8
  %v4 = call i32 @f0(i64 %v3)
  ret i32 %v4
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
