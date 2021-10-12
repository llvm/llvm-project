; RUN: opt -S -globaldce < %s | FileCheck %s

;
; (1) a dead global referenced in @llvm.used, but marked as conditionally used
; by both @condition_live and @condition_dead, with an *ANY* type, is alive
;
define internal void @func_conditionally_by_two_conditions_any() {
  ; CHECK: @func_conditionally_by_two_conditions_any
  ret void
}

;
; (2) a dead global referenced in @llvm.used, but marked as conditionally used
; by both @condition_live and @condition_dead, with an *ALL* type, is removed
;
define internal void @func_conditionally_by_two_conditions_all() {
  ; CHECK-NOT: @func_conditionally_by_two_conditions_all
  ret void
}

@condition_live = internal unnamed_addr constant i64 42
@condition_dead = internal unnamed_addr constant i64 42

@llvm.used = appending global [3 x i8*] [
   i8* bitcast (i64* @condition_live to i8*),
   i8* bitcast (void ()* @func_conditionally_by_two_conditions_any to i8*),
   i8* bitcast (void ()* @func_conditionally_by_two_conditions_all to i8*)
], section "llvm.metadata"

!1 = !{void ()* @func_conditionally_by_two_conditions_any, i32 0, !{ i64* @condition_live, i64* @condition_dead } }
!2 = !{void ()* @func_conditionally_by_two_conditions_all, i32 1, !{ i64* @condition_live, i64* @condition_dead } }
!llvm.used.conditional = !{!1, !2}
