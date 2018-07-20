; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = add(##g0,mpyi(r{{[0-9]+}},#24))

%s.0 = type { i32, i32, i32, i32, i32, i8 }

@g0 = common global [2 x %s.0] zeroinitializer, align 8

declare void @f0(%s.0*)

; Function Attrs: nounwind readnone
define void @f1(i32 %a0) #0 {
b0:
  %v0 = getelementptr inbounds [2 x %s.0], [2 x %s.0]* @g0, i32 0, i32 %a0
  call void @f0(%s.0* %v0) #1
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
