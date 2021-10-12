; RUN: opt -S -globaldce < %s | FileCheck %s

;
; (1) a regular dead global referenced in @llvm.used is kept alive
;
define internal void @func_marked_as_used() {
  ; CHECK: @func_marked_as_used
  ret void
}

;
; (2) a dead global referenced in @llvm.used, but marked as conditionally used
; in !llvm.used.conditional where the condition is alive, is kept alive
;
define internal void @func_conditionally_used_and_live() {
  ; CHECK: @func_conditionally_used_and_live
  ret void
}

;
; (3) a dead global referenced in @llvm.used, but marked as conditionally used
; in !llvm.used.conditional where the condition is dead, is eliminated
;
define internal void @func_conditionally_used_and_dead() {
  ; CHECK-NOT: @func_conditionally_used_and_dead
  ret void
}

@condition_live = internal unnamed_addr constant i64 42
@condition_dead = internal unnamed_addr constant i64 42

@llvm.used = appending global [4 x i8*] [
   i8* bitcast (void ()* @func_marked_as_used to i8*),
   i8* bitcast (i64* @condition_live to i8*),
   i8* bitcast (void ()* @func_conditionally_used_and_live to i8*),
   i8* bitcast (void ()* @func_conditionally_used_and_dead to i8*)
], section "llvm.metadata"

!1 = !{void ()* @func_conditionally_used_and_live, i32 0, !{i64* @condition_live}}
!2 = !{void ()* @func_conditionally_used_and_dead, i32 0, !{i64* @condition_dead}}
!llvm.used.conditional = !{!1, !2}
