; RUN: opt -passes=globalopt -S -o - < %s | FileCheck %s

@glbl = internal global i8* null

define void @test1a() {
; CHECK-LABEL: @test1a(
; CHECK-NOT: store
; CHECK-NEXT: ret void
  store i8* null, i8** @glbl
  ret void
}

define void @test1b(i8* %p) {
; CHECK-LABEL: @test1b(
; CHECK-NEXT: store
; CHECK-NEXT: ret void
  store i8* %p, i8** @glbl
  ret void
}

define void @test2() {
; CHECK-LABEL: @test2(
; CHECK: alloca i8
  %txt = alloca i8
  call void @foo2(i8* %txt)
  %call2 = call i8* @strdup(i8* %txt)
  store i8* %call2, i8** @glbl
  ret void
}
declare i8* @strdup(i8*)
declare void @foo2(i8*)

define void @test3() uwtable personality i32 (i32, i64, i8*, i8*)* @__gxx_personality_v0 {
; CHECK-LABEL: @test3(
; CHECK-NOT: bb1:
; CHECK-NOT: bb2:
; CHECK: invoke
  %ptr = invoke i8* @_Znwm(i64 1)
          to label %bb1 unwind label %bb2
bb1:
  store i8* %ptr, i8** @glbl
  unreachable
bb2:
  %tmp1 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %tmp1
}
declare i32 @__gxx_personality_v0(i32, i64, i8*, i8*)
declare i8* @_Znwm(i64)
