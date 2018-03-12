; RUN: llc -march=hexagon  < %s | FileCheck %s
; CHECK: callr {{r[0-9]+}}

%s.0 = type { [1 x %s.1], [4 x i8*] }
%s.1 = type { [1 x %s.2], i32, [4 x i8] }
%s.2 = type { [16 x i32] }

; Function Attrs: noreturn nounwind
define hidden void @f0() #0 section ".text.compat" {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  call void undef(%s.0* undef) #1
  unreachable
}

attributes #0 = { noreturn nounwind }
attributes #1 = { noreturn }
