; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; A virtual call should keep a virtual function alive only if the vtable it is loaded from is alive too.

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)
declare { ptr, i1 } @llvm.type.checked.load.relative(ptr, i32, metadata)

; @vt.live is used by @test, @f.live is called through it and kept.
; CHECK: @vt.live = internal constant [1 x ptr] [ptr @f.live]
@vt.live = internal constant [1 x ptr] [ptr @f.live], !type !0, !vcall_visibility !2

; @vt.dead is unused by @test, @f.dead is eliminated.
; CHECK-NOT: @vt.dead =
@vt.dead = internal constant [1 x ptr] [ptr @f.dead], !type !0, !vcall_visibility !2

; @vt.late is only used by @f.live, so it is live but it takes more steps to figure that out.
; CHECK: @vt.late = internal constant [1 x ptr] [ptr @f.late]
@vt.late = internal constant [1 x ptr] [ptr @f.late], !type !1, !vcall_visibility !2

; @vt.public has a public vcall_visibility (encoded as 0), and is therefore not safe for VFE.
; So @f.public is kept even though it is never called in this module (virtually or otherwise).
; CHECK: @vt.public = internal constant [1 x ptr] [ptr @f.public]
@vt.public = internal constant [1 x ptr] [ptr @f.public], !type !4, !vcall_visibility !5

; Model a C++ vtable, which is only used through the alias @vt.rel.
; @vt.rel.local becomes live only through the alias, and @f.rel must be kept.
; CHECK: @vt.rel.local = private unnamed_addr constant { [2 x i32] }
; CHECK-SAME: { [2 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @f.rel1 to i64), i64 ptrtoint (ptr @vt.rel.local to i64)) to i32), i32 0] }
@vt.rel.local = private unnamed_addr constant { [2 x i32] } { [2 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @f.rel1 to i64), i64 ptrtoint (ptr @vt.rel.local to i64)) to i32),
  i32 trunc (i64 sub (i64 ptrtoint (ptr @f.rel2 to i64), i64 ptrtoint (ptr @vt.rel.local to i64)) to i32)
]}, align 4, !type !6, !vcall_visibility !2

; CHECK: @vt.rel = internal unnamed_addr alias { [2 x i32] }, ptr @vt.rel.local
@vt.rel = internal unnamed_addr alias { [2 x i32] }, ptr @vt.rel.local

; CHECK: define internal void @f.live(
define internal void @f.live(ptr %obj) {
  store ptr @vt.late, ptr %obj
  ret void
}

; CHECK-NOT: define internal void @f.dead(
define internal void @f.dead(ptr %obj) {
  ret void
}

; CHECK: define internal void @f.late(
define internal void @f.late(ptr %obj) {
  ret void
}

; CHECK: define internal i32 @f.rel1(
define internal i32 @f.rel1(ptr %obj) {
  ret i32 1
}

; CHECK-NOT: define internal i32 @f.rel2(
define internal i32 @f.rel2(ptr %obj) {
  ret i32 2
}

define void @test(ptr %obj) {
  store ptr @vt.live, ptr %obj
  %pair = call { ptr, i1 } @llvm.type.checked.load(ptr @vt.live, i32 0, metadata !"typeid")
  %fptr = extractvalue { ptr, i1 } %pair, 0
  call void %fptr(ptr %obj)
  %vtable = load ptr, ptr %obj
  %pair.late = call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid.late")
  %fptr.late = extractvalue { ptr, i1 } %pair.late, 0
  call void %fptr.late(ptr %obj)
  ret void
}

; This function is part of a vcall_visibility public vtable, and must be kept.
; CHECK: define internal void @f.public(
define internal void @f.public(ptr %obj) {
  ret void
}

; Make the vtable escape.
define void @test.public(ptr %obj) {
  store ptr @vt.public, ptr %obj
  ret void
}

define void @test.rel(ptr %obj) {
  store ptr @vt.rel, ptr %obj
  %vtable = load ptr, ptr %obj
  %pair = call { ptr, i1 } @llvm.type.checked.load.relative(ptr %vtable, i32 0, metadata !"typeid.rel")
  %fptr = extractvalue { ptr, i1 } %pair, 0
  call void %fptr(ptr %obj)
  ret void
}

!llvm.module.flags = !{!3}

!0 = !{i64 0, !"typeid"}
!1 = !{i64 0, !"typeid.late"}
!2 = !{i64 2}
!3 = !{i32 1, !"Virtual Function Elim", i32 1}
!4 = !{i64 0, !"typeid.public"}
!5 = !{i64 0}
!6 = !{i64 0, !"typeid.rel"}
