; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: callr {{r[0-9]+}}

%s.0 = type { [1 x %s.1], [4 x ptr] }
%s.1 = type { [1 x %s.2], i32, [4 x i8] }
%s.2 = type { [16 x i32] }

define hidden void @f0(ptr %a0) #0 {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  ret void

b2:                                               ; preds = %b0
  call void %a0(ptr null) #1
  unreachable
}
 
attributes #0 = { noreturn nounwind }
attributes #1 = { noreturn }
