; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: vaddh(r{{[0-9]+}},r{{[0-9]+}})

@g0 = external global i32
@g1 = external global i32

define void @f0() #0 {
b0:
  %v0 = load i32, ptr @g0, align 4
  %v1 = load i32, ptr @g1, align 4
  %v2 = call i32 @llvm.hexagon.A2.svaddh(i32 %v0, i32 %v1)
  store i32 %v2, ptr @g1, align 4
  ret void
}

declare i32 @llvm.hexagon.A2.svaddh(i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv5" }
