; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='require<globals-aa>,dse' -S | FileCheck %s
;
; The LCG does not reach internal resolvers of unused ifuncs, internal
; personality functions, or their callees. @helper reads @T, so it must not look
; readnone there just because the SCC walk skipped it.

@T = internal global i32 0
@f = ifunc void (), ptr @resolver

define internal void @helper() nounwind {
  %v = load i32, ptr @T
  ret void
}

define internal ptr @resolver() {
; CHECK-LABEL: define internal ptr @resolver(
; CHECK: store i32 1, ptr @T
; CHECK-NEXT: call void @helper()
; CHECK-NEXT: store i32 2, ptr @T
  store i32 1, ptr @T
  call void @helper()
  store i32 2, ptr @T
  ret ptr null
}

define internal i32 @pers() {
; CHECK-LABEL: define internal i32 @pers(
; CHECK: store i32 3, ptr @T
; CHECK-NEXT: call void @helper()
; CHECK-NEXT: store i32 4, ptr @T
  store i32 3, ptr @T
  call void @helper()
  store i32 4, ptr @T
  ret i32 0
}

define void @user() personality ptr @pers {
  ret void
}
