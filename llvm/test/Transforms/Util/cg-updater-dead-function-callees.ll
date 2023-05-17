; RUN: opt -passes=inline,attributor-cgscc,tailcallelim -S %s | FileCheck %s
;
; CHECK: define void @foo()
; CHECK: declare i32 @baz()
; CHECK-NOT: void @goo()
; CHECK-NOT: void @bar()

define void @foo() {
  call fastcc void @bar()
  ret void
}

define internal fastcc void @goo() {
  call fastcc void @bar()
  ret void
}

define internal fastcc void @bar() {
  %call = call i32 @baz()
  %cond = icmp eq i32 %call, 0
  br i1 %cond, label %if.then, label %if.end

if.then:
  call fastcc void @goo()
  br label %if.end

if.end:
  ret void
}

declare i32 @baz()
