; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-NOT: pcrelR0

target triple = "hexagon"

%s.0 = type { i32, i32 }

@g0 = global %s.0 zeroinitializer, align 4

@e0 = alias void (%s.0*, i32, i32), void (%s.0*, i32, i32)* @f0

; Function Attrs: nounwind
define void @f0(%s.0* %a0, i32 %a1, i32 %a2) unnamed_addr #0 align 2 {
b0:
  %v0 = alloca %s.0*, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  store %s.0* %a0, %s.0** %v0, align 4
  store i32 %a1, i32* %v1, align 4
  store i32 %a2, i32* %v2, align 4
  %v3 = load %s.0*, %s.0** %v0
  %v4 = getelementptr inbounds %s.0, %s.0* %v3, i32 0, i32 0
  %v5 = load i32, i32* %v2, align 4
  store i32 %v5, i32* %v4, align 4
  %v6 = getelementptr inbounds %s.0, %s.0* %v3, i32 0, i32 1
  %v7 = load i32, i32* %v1, align 4
  store i32 %v7, i32* %v6, align 4
  ret void
}

define internal void @f1() {
b0:
  call void @e0(%s.0* @g0, i32 3, i32 7)
  ret void
}

; Function Attrs: nounwind
define i32 @f2() #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 0, i32* %v0
  ret i32 0
}

define internal void @f3() {
b0:
  call void @f1()
  ret void
}

attributes #0 = { nounwind }
