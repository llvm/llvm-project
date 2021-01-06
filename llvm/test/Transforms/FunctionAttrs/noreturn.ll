; RUN: opt < %s -passes='function-attrs' -S | FileCheck %s

declare i32 @f()

; CHECK: Function Attrs: noreturn
; CHECK-NEXT: @noreturn()
declare i32 @noreturn() noreturn

; CHECK: Function Attrs: noreturn
; CHECK-NEXT: @caller()
define i32 @caller() {
  %c = call i32 @noreturn()
  ret i32 %c
}

; CHECK: Function Attrs: noreturn
; CHECK-NEXT: @caller2()
define i32 @caller2() {
  %c = call i32 @caller()
  ret i32 %c
}

; CHECK: Function Attrs: noreturn
; CHECK-NEXT: @caller3()
define i32 @caller3() {
entry:
  br label %end
end:
  %c = call i32 @noreturn()
  ret i32 %c
}

; CHECK-NOT: Function Attrs: {{.*}}noreturn
; CHECK: define i32 @caller4()
define i32 @caller4() {
entry:
  br label %end
end:
  %c = call i32 @f()
  ret i32 %c
}

; CHECK-NOT: Function Attrs: {{.*}}noreturn
; CHECK: @caller5()
; We currently don't handle unreachable blocks.
define i32 @caller5() {
entry:
  %c = call i32 @noreturn()
  ret i32 %c
unreach:
  %d = call i32 @f()
  ret i32 %d
}

; CHECK-NOT: Function Attrs: {{.*}}noreturn
; CHECK: @caller6()
define i32 @caller6() naked {
  %c = call i32 @noreturn()
  ret i32 %c
}

; CHECK: Function Attrs: {{.*}}noreturn
; CHECK-NEXT: @alreadynoreturn()
define i32 @alreadynoreturn() noreturn {
  unreachable
}
