; RUN: opt -disable-output -passes=helloworld %s 2>&1 | FileCheck %s

; CHECK: {{^}}foo{{$}}
define i32 @foo() {
  %a = add i32 2, 3
  ret i32 %a
}

; CHECK-NEXT: {{^}}bar{{$}}
define void @bar() {
  ret void
}
