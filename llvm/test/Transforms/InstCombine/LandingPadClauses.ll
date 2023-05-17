; RUN: opt < %s -passes=instcombine -S | FileCheck %s

@T1 = external constant i32
@T2 = external constant i32
@T3 = external constant i32

declare i32 @generic_personality(i32, i64, ptr, ptr)
declare i32 @__gxx_personality_v0(i32, i64, ptr, ptr)
declare i32 @__objc_personality_v0(i32, i64, ptr, ptr)
declare i32 @__C_specific_handler(...)

declare void @bar()

define void @foo_generic() personality ptr @generic_personality {
; CHECK-LABEL: @foo_generic(
  invoke void @bar()
    to label %cont.a unwind label %lpad.a
cont.a:
  invoke void @bar()
    to label %cont.b unwind label %lpad.b
cont.b:
  invoke void @bar()
    to label %cont.c unwind label %lpad.c
cont.c:
  invoke void @bar()
    to label %cont.d unwind label %lpad.d
cont.d:
  invoke void @bar()
    to label %cont.e unwind label %lpad.e
cont.e:
  invoke void @bar()
    to label %cont.f unwind label %lpad.f
cont.f:
  invoke void @bar()
    to label %cont.g unwind label %lpad.g
cont.g:
  invoke void @bar()
    to label %cont.h unwind label %lpad.h
cont.h:
  invoke void @bar()
    to label %cont.i unwind label %lpad.i
cont.i:
  ret void

lpad.a:
  %a = landingpad { ptr, i32 }
          catch ptr @T1
          catch ptr @T2
          catch ptr @T1
          catch ptr @T2
  unreachable
; CHECK: %a = landingpad
; CHECK-NEXT: @T1
; CHECK-NEXT: @T2
; CHECK-NEXT: unreachable

lpad.b:
  %b = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
          catch ptr @T1
  unreachable
; CHECK: %b = landingpad
; CHECK-NEXT: filter
; CHECK-NEXT: unreachable

lpad.c:
  %c = landingpad { ptr, i32 }
          catch ptr @T1
          filter [1 x ptr] [ptr @T1]
          catch ptr @T2
  unreachable
; Caught types should not be removed from filters
; CHECK: %c = landingpad
; CHECK-NEXT: catch ptr @T1
; CHECK-NEXT: filter [1 x ptr] [ptr @T1]
; CHECK-NEXT: catch ptr @T2 
; CHECK-NEXT: unreachable

lpad.d:
  %d = landingpad { ptr, i32 }
          filter [3 x ptr] zeroinitializer
  unreachable
; CHECK: %d = landingpad
; CHECK-NEXT: filter [1 x ptr] zeroinitializer
; CHECK-NEXT: unreachable

lpad.e:
  %e = landingpad { ptr, i32 }
          catch ptr @T1
          filter [3 x ptr] [ptr @T1, ptr @T2, ptr @T2]
  unreachable
; Caught types should not be removed from filters
; CHECK: %e = landingpad
; CHECK-NEXT: catch ptr @T1
; CHECK-NEXT: filter [2 x ptr] [ptr @T1, ptr @T2]
; CHECK-NEXT: unreachable

lpad.f:
  %f = landingpad { ptr, i32 }
          filter [2 x ptr] [ptr @T2, ptr @T1]
          filter [1 x ptr] [ptr @T1]
  unreachable
; CHECK: %f = landingpad
; CHECK-NEXT: filter [1 x ptr] [ptr @T1]
; CHECK-NEXT: unreachable

lpad.g:
  %g = landingpad { ptr, i32 }
          filter [1 x ptr] [ptr @T1]
          catch ptr @T3
          filter [2 x ptr] [ptr @T2, ptr @T1]
  unreachable
; CHECK: %g = landingpad
; CHECK-NEXT: filter [1 x ptr] [ptr @T1]
; CHECK-NEXT: catch ptr @T3
; CHECK-NEXT: unreachable

lpad.h:
  %h = landingpad { ptr, i32 }
          filter [2 x ptr] [ptr @T1, ptr null]
          filter [1 x ptr] zeroinitializer
  unreachable
; CHECK: %h = landingpad
; CHECK-NEXT: filter [1 x ptr] zeroinitializer
; CHECK-NEXT: unreachable

lpad.i:
  %i = landingpad { ptr, i32 }
          cleanup
          filter [0 x ptr] zeroinitializer
  unreachable
; CHECK: %i = landingpad
; CHECK-NEXT: filter
; CHECK-NEXT: unreachable
}

define void @foo_cxx() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @foo_cxx(
  invoke void @bar()
    to label %cont.a unwind label %lpad.a
cont.a:
  invoke void @bar()
    to label %cont.b unwind label %lpad.b
cont.b:
  invoke void @bar()
    to label %cont.c unwind label %lpad.c
cont.c:
  invoke void @bar()
    to label %cont.d unwind label %lpad.d
cont.d:
  ret void

lpad.a:
  %a = landingpad { ptr, i32 }
          catch ptr null
          catch ptr @T1
  unreachable
; CHECK: %a = landingpad
; CHECK-NEXT: null
; CHECK-NEXT: unreachable

lpad.b:
  %b = landingpad { ptr, i32 }
          filter [1 x ptr] zeroinitializer
  unreachable
; CHECK: %b = landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: unreachable

lpad.c:
  %c = landingpad { ptr, i32 }
          filter [2 x ptr] [ptr @T1, ptr null]
  unreachable
; CHECK: %c = landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: unreachable

lpad.d:
  %d = landingpad { ptr, i32 }
          cleanup
          catch ptr null
  unreachable
; CHECK: %d = landingpad
; CHECK-NEXT: null
; CHECK-NEXT: unreachable
}

define void @foo_objc() personality ptr @__objc_personality_v0 {
; CHECK-LABEL: @foo_objc(
  invoke void @bar()
    to label %cont.a unwind label %lpad.a
cont.a:
  invoke void @bar()
    to label %cont.b unwind label %lpad.b
cont.b:
  invoke void @bar()
    to label %cont.c unwind label %lpad.c
cont.c:
  invoke void @bar()
    to label %cont.d unwind label %lpad.d
cont.d:
  ret void

lpad.a:
  %a = landingpad { ptr, i32 }
          catch ptr null
          catch ptr @T1
  unreachable
; CHECK: %a = landingpad
; CHECK-NEXT: null
; CHECK-NEXT: unreachable

lpad.b:
  %b = landingpad { ptr, i32 }
          filter [1 x ptr] zeroinitializer
  unreachable
; CHECK: %b = landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: unreachable

lpad.c:
  %c = landingpad { ptr, i32 }
          filter [2 x ptr] [ptr @T1, ptr null]
  unreachable
; CHECK: %c = landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: unreachable

lpad.d:
  %d = landingpad { ptr, i32 }
          cleanup
          catch ptr null
  unreachable
; CHECK: %d = landingpad
; CHECK-NEXT: null
; CHECK-NEXT: unreachable
}

define void @foo_seh() personality ptr @__C_specific_handler {
; CHECK-LABEL: @foo_seh(
  invoke void @bar()
    to label %cont.a unwind label %lpad.a
cont.a:
  invoke void @bar()
    to label %cont.b unwind label %lpad.b
cont.b:
  invoke void @bar()
    to label %cont.c unwind label %lpad.c
cont.c:
  invoke void @bar()
    to label %cont.d unwind label %lpad.d
cont.d:
  ret void

lpad.a:
  %a = landingpad { ptr, i32 }
          catch ptr null
          catch ptr @T1
  unreachable
; CHECK: %a = landingpad
; CHECK-NEXT: null
; CHECK-NEXT: unreachable

lpad.b:
  %b = landingpad { ptr, i32 }
          filter [1 x ptr] zeroinitializer
  unreachable
; CHECK: %b = landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: unreachable

lpad.c:
  %c = landingpad { ptr, i32 }
          filter [2 x ptr] [ptr @T1, ptr null]
  unreachable
; CHECK: %c = landingpad
; CHECK-NEXT: cleanup
; CHECK-NEXT: unreachable

lpad.d:
  %d = landingpad { ptr, i32 }
          cleanup
          catch ptr null
  unreachable
; CHECK: %d = landingpad
; CHECK-NEXT: null
; CHECK-NEXT: unreachable
}
