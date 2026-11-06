; RUN: opt < %s -aa-pipeline=globals-aa -passes='require<globals-aa>,dse' -S | FileCheck %s
;
; A broker modeled as having no effects may still invoke its callback, so the
; callback's effects must reach the caller. Callees are nounwind because
; may-throw calls keep stores regardless of AA.

@X = internal global i32 0
@Y = internal global i32 0
@A = internal global i32 0
@B = internal global i32 0

declare !callback !0 void @broker(ptr) readnone nounwind
declare void @sink(ptr) readnone nounwind

define void @reads_x() nounwind {
  %v = load i32, ptr @X
  ret void
}

define void @calls_reads_x() nounwind {
  call void @broker(ptr @reads_x)
  ret void
}

define void @test_callback() {
; CHECK-LABEL: define void @test_callback(
; CHECK: store i32 1, ptr @X
; CHECK-NEXT: call void @calls_reads_x()
; CHECK-NEXT: store i32 2, ptr @X
  store i32 1, ptr @X
  call void @calls_reads_x()
  store i32 2, ptr @X
  ret void
}

define void @pure() nounwind {
  ret void
}

define void @calls_pure() nounwind {
  call void @broker(ptr @pure)
  ret void
}

define void @test_pure_callback() {
; CHECK-LABEL: define void @test_pure_callback(
; CHECK-NOT: store i32 1, ptr @Y
; CHECK: call void @calls_pure()
; CHECK-NEXT: store i32 2, ptr @Y
  store i32 1, ptr @Y
  call void @calls_pure()
  store i32 2, ptr @Y
  ret void
}

; @user and @reader reference each other, forming one RefSCC of two SCCs. @user
; is visited first, while @reader only has partial info without the read of @B.
; This relies on the definition order below.
define void @test_unvisited_callback() {
; CHECK-LABEL: define void @test_unvisited_callback(
; CHECK: store i32 1, ptr @B
; CHECK-NEXT: call void @user()
; CHECK-NEXT: store i32 2, ptr @B
  store i32 1, ptr @B
  call void @user()
  store i32 2, ptr @B
  ret void
}

define void @user() nounwind {
  call void @broker(ptr @reader)
  ret void
}

define void @reader() nounwind {
  %v = load i32, ptr @A
  call void @reads_b()
  call void @sink(ptr @user)
  ret void
}

define void @reads_b() nounwind {
  %v = load i32, ptr @B
  ret void
}

!0 = !{!1}
!1 = !{i64 0, i64 -1, i1 false}
