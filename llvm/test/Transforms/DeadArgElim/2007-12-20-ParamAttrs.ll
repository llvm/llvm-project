; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

%struct = type { }

@g = global i8 0

; CHECK: define internal void @foo(i8 signext %y) [[NUW:#[0-9]+]]

define internal zeroext i8 @foo(i8* inreg %p, i8 signext %y, ... )  nounwind {
  store i8 %y, i8* @g
  ret i8 0
}

define i32 @bar() {
; CHECK: call void @foo(i8 signext 1) [[NUW]]
  %A = call zeroext i8(i8*, i8, ...) @foo(i8* inreg null, i8 signext 1, %struct* byval(%struct) null ) nounwind
  ret i32 0
}

; CHECK: attributes [[NUW]] = { nounwind }
