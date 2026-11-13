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

; FIXME: The metadata of a call is dropped along with its kcfi operand bundle.
; CHECK-LABEL: define void @h() #0 personality ptr @__gxx_personality_v0
define void @h() #0 personality ptr @__gxx_personality_v0 {
  ; CHECK: call void @f1(){{$}}
  call void @f1() [ "kcfi"(i32 10) ], !prof !0, !callsite !1

  ; CHECK: call ptr @f3(){{$}}
  call ptr @f3() [ "kcfi"(i32 12) ], !memprof !2, !callsite !1

  ; CHECK: invoke void @f2()
  ; CHECK-NEXT: to label %ok unwind label %lpad{{$}}
  invoke void @f2() [ "kcfi"(i32 11) ]
          to label %ok unwind label %lpad, !prof !0

ok:
  ret void

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %lp
}

declare ptr @f3() #0 prefix i32 12

declare i32 @__gxx_personality_v0(...)

attributes #0 = { "kcfi-target" }

!0 = !{!"branch_weights", i32 1000}
!1 = !{i64 123}
!2 = !{!3}
!3 = !{!4, !"notcold"}
!4 = !{i64 123, i64 456}
