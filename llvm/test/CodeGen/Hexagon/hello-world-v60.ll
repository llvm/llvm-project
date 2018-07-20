; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: call puts

@g0 = private unnamed_addr constant [13 x i8] c"Hello World!\00"

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = tail call i32 @puts(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @g0, i32 0, i32 0))
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture) #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
