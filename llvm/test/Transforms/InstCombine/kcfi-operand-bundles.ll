; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define void @f1() #0 prefix i32 10 {
  ret void
}

declare void @f2() #0 prefix i32 11

; CHECK-LABEL: define void @g(ptr noundef %x) #0
define void @g(ptr noundef %x) #0 {
  ; CHECK: call void %x() [ "kcfi"(i32 10) ]
  call void %x() [ "kcfi"(i32 10) ]

  ; COM: Must drop the kcfi operand bundle from direct calls.
  ; CHECK: call void @f1()
  ; CHECK-NOT: [ "kcfi"(i32 10) ]
  call void @f1() [ "kcfi"(i32 10) ]

  ; CHECK: call void @f2()
  ; CHECK-NOT: [ "kcfi"(i32 10) ]
  call void @f2() [ "kcfi"(i32 10) ]
  ret void
}

attributes #0 = { "kcfi-target" }
