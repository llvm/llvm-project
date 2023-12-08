; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

%struct = type { }

@g = global i8 0

; CHECK: define internal void @foo(i8 signext %y) [[NUW:#[0-9]+]]

define internal zeroext i8 @foo(ptr inreg %p, i8 signext %y, ... )  nounwind {
  store i8 %y, ptr @g
  ret i8 0
}

define i32 @bar() {
; CHECK: call void @foo(i8 signext 1) [[NUW]]
  %A = call zeroext i8(ptr, i8, ...) @foo(ptr inreg null, i8 signext 1, ptr byval(%struct) null ) nounwind
  ret i32 0
}

; CHECK: attributes [[NUW]] = { nounwind }
