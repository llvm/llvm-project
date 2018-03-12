; RUN: llc -march=hexagon < %s | FileCheck %s
; Honor the alignment of a halfword on byte boundaries.
; CHECK-NOT: memh

target triple = "hexagon-unknown-linux-gnu"

%s.0 = type <{ i16, i8, i16 }>

@g0 = common global %s.0 zeroinitializer, align 1

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 %a0, i32* %v0, align 4
  store i8 1, i8* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 1), align 1
  %v1 = load i16, i16* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 2), align 1
  %v2 = add i16 %v1, 1
  store i16 %v2, i16* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 2), align 1
  %v3 = load i32, i32* %v0, align 4
  %v4 = icmp ne i32 %v3, 0
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = load i16, i16* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 2), align 1
  %v6 = zext i16 %v5 to i32
  %v7 = or i32 %v6, 6144
  %v8 = trunc i32 %v7 to i16
  store i16 %v8, i16* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 2), align 1
  br label %b3

b2:                                               ; preds = %b0
  %v9 = load i16, i16* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 2), align 1
  %v10 = zext i16 %v9 to i32
  %v11 = or i32 %v10, 2048
  %v12 = trunc i32 %v11 to i16
  store i16 %v12, i16* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 2), align 1
  br label %b3

b3:                                               ; preds = %b2, %b1
  ret i32 0
}

attributes #0 = { nounwind }
