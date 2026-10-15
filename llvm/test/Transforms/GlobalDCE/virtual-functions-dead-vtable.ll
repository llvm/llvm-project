; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; A virtual call should keep a virtual function alive only if the vtable it is loaded from is alive too.

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

; @vt.live is used by @test, @f.live is called through it and kept.
; CHECK: @vt.live = internal constant [1 x ptr] [ptr @f.live]
@vt.live = internal constant [1 x ptr] [ptr @f.live], !type !0, !vcall_visibility !2

; @vt.dead is unused by @test, @f.dead is eliminated.
; CHECK-NOT: @vt.dead =
@vt.dead = internal constant [1 x ptr] [ptr @f.dead], !type !0, !vcall_visibility !2

; @vt.late is only used by @f.live, so it is live but it takes more steps to figure that out.
; CHECK: @vt.late = internal constant [1 x ptr] [ptr @f.late]
@vt.late = internal constant [1 x ptr] [ptr @f.late], !type !1, !vcall_visibility !2

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

!llvm.module.flags = !{!3}

!0 = !{i64 0, !"typeid"}
!1 = !{i64 0, !"typeid.late"}
!2 = !{i64 2}
!3 = !{i32 1, !"Virtual Function Elim", i32 1}
