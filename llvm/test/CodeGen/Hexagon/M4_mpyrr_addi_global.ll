; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = add(##g0{{.*}},mpyi(r{{[0-9]+}},r{{[0-9]+}}))

%s.0 = type { %s.1, %s.1* }
%s.1 = type { i8, i8, i8, i8, i16, i16, i8, [3 x i8], [20 x %s.2] }
%s.2 = type { i8, i8, [2 x i8], [2 x i8] }

@g0 = external global [2 x %s.0]

declare void @f0(%s.1**)

; Function Attrs: nounwind readnone
define void @f1(i32 %a0) #0 {
b0:
  %v0 = getelementptr inbounds [2 x %s.0], [2 x %s.0]* @g0, i32 0, i32 %a0
  %v1 = getelementptr inbounds %s.0, %s.0* %v0, i32 0, i32 1
  call void @f0(%s.1** %v1) #1
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
