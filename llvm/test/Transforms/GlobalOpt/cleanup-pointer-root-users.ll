; RUN: opt -passes=globalopt -S -o - < %s | FileCheck %s

@glbl = internal global ptr null

define void @test1a() {
; CHECK-LABEL: @test1a(
; CHECK-NOT: store
; CHECK-NEXT: ret void
  store ptr null, ptr @glbl
  ret void
}

define void @test1b(ptr %p) {
; CHECK-LABEL: @test1b(
; CHECK-NEXT: store
; CHECK-NEXT: ret void
  store ptr %p, ptr @glbl
  ret void
}

define void @test2() {
; CHECK-LABEL: @test2(
; CHECK: alloca i8
  %txt = alloca i8
  call void @foo2(ptr %txt)
  %call2 = call ptr @strdup(ptr %txt)
  store ptr %call2, ptr @glbl
  ret void
}
declare ptr @strdup(ptr)
declare void @foo2(ptr)

define void @test3() uwtable personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @test3(
; CHECK-NOT: bb1:
; CHECK-NOT: bb2:
; CHECK: invoke
  %ptr = invoke ptr @_Znwm(i64 1)
          to label %bb1 unwind label %bb2
bb1:
  store ptr %ptr, ptr @glbl
  unreachable
bb2:
  %tmp1 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %tmp1
}
declare i32 @__gxx_personality_v0(i32, i64, ptr, ptr)
declare ptr @_Znwm(i64)
