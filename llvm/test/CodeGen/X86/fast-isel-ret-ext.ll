; RUN: llc < %s  -O0 -fast-isel-abort=1 -mtriple i686-apple-darwin10 | FileCheck %s
; RUN: llc < %s  -O0 -fast-isel-abort=1 -mtriple x86_64-apple-darwin10 | FileCheck %s

define zeroext i8 @test1(i32 %y) nounwind {
  %conv = trunc i32 %y to i8
  ret i8 %conv
  ; CHECK-LABEL: test1:
  ; CHECK: movzbl {{.*}}, %eax
}

define signext i8 @test2(i32 %y) nounwind {
  %conv = trunc i32 %y to i8
  ret i8 %conv
  ; CHECK-LABEL: test2:
  ; CHECK: movsbl {{.*}}, %eax
}

define zeroext i16 @test3(i32 %y) nounwind {
  %conv = trunc i32 %y to i16
  ret i16 %conv
  ; CHECK-LABEL: test3:
  ; CHECK: movzwl {{.*}}, %eax
}

define signext i16 @test4(i32 %y) nounwind {
  %conv = trunc i32 %y to i16
  ret i16 %conv
  ; CHECK-LABEL: test4:
  ; CHECK: {{(movswl.%di, %eax|cwtl)}}
}

define zeroext i1 @test5(i32 %y) nounwind {
  %conv = trunc i32 %y to i1
  ret i1 %conv
  ; CHECK-LABEL: test5:
  ; CHECK: andb $1
  ; CHECK-NEXT: ret
}
