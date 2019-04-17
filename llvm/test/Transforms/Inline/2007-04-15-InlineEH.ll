; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; PR1335

target triple = "i686-pc-linux-gnu"

declare i32 @__gxx_personality_v0(...)

declare void @a()

declare void @b()

declare void @c()

define void @f() {
; CHECK-LABEL: define void @f()
entry:
  call void asm "rdtsc\0A\09movl %eax, $0\0A\09movl %edx, $1", "=*imr,=*imr,~{dirflag},~{fpsr},~{flags},~{dx},~{ax}"( i32* null, i32* null ) nounwind
; CHECK: call void asm
  unreachable
}

define void @g() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: define void @g() personality i32 (...)* @__gxx_personality_v0
entry:
  invoke void @a() to label %invcont1 unwind label %cleanup
; CHECK-NOT: {{call|invoke}}
; CHECK: invoke void @a()

invcont1:
  invoke void @b() to label %invcont2 unwind label %cleanup
; CHECK-NOT: {{call|invoke}}
; CHECK: invoke void @b()

invcont2:
  invoke void @c() to label %invcont3 unwind label %cleanup
; CHECK-NOT: {{call|invoke}}
; CHECK: invoke void @c()

invcont3:
  invoke void @f() to label %invcont4 unwind label %cleanup
; CHECK-NOT: {{call|invoke}}
; CHECK: call void asm
; CHECK-NOT: {{call|invoke}}

invcont4:
  ret void

cleanup:
  %ex = landingpad {i8*, i32} cleanup
  resume { i8*, i32 } %ex
}

define void @h() {
; CHECK-LABEL: define void @h() personality i32 (...)* @__gxx_personality_v0
entry:
  call void @g()
; CHECK-NOT: {{call|invoke}}
; CHECK: invoke void @a()
; CHECK-NOT: {{call|invoke}}
; CHECK: invoke void @b()
; CHECK-NOT: {{call|invoke}}
; CHECK: invoke void @c()
; CHECK-NOT: {{call|invoke}}
; CHECK: call void asm
; CHECK-NOT: {{call|invoke}}

  ret void
}
