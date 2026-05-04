; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK-NOT: pcrelR0

target triple = "hexagon"

%s.0 = type { i32, i32 }

@g0 = global %s.0 zeroinitializer, align 4

@e0 = alias void (ptr, i32, i32), ptr @f0

; Function Attrs: nounwind
define void @f0(ptr %a0, i32 %a1, i32 %a2) unnamed_addr #0 align 2 {
b0:
  %v0 = alloca ptr, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  store ptr %a0, ptr %v0, align 4
  store i32 %a1, ptr %v1, align 4
  store i32 %a2, ptr %v2, align 4
  %v3 = load ptr, ptr %v0
  %v5 = load i32, ptr %v2, align 4
  store i32 %v5, ptr %v3, align 4
  %v6 = getelementptr inbounds %s.0, ptr %v3, i32 0, i32 1
  %v7 = load i32, ptr %v1, align 4
  store i32 %v7, ptr %v6, align 4
  ret void
}

define internal void @f1() {
b0:
  call void @e0(ptr @g0, i32 3, i32 7)
  ret void
}

; Function Attrs: nounwind
define i32 @f2() #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 0, ptr %v0
  ret i32 0
}

define internal void @f3() {
b0:
  call void @f1()
  ret void
}

attributes #0 = { nounwind }
