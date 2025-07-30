; RUN: opt -passes='loop-mssa(loop-instsimplify,loop-simplifycfg,licm,simple-loop-unswitch)' -enable-nontrivial-unswitch -verify-memoryssa %s -disable-output
; REQUIRES: asserts

%0 = type { i24, i32, i32, i8 }

@g_1077 = external dso_local local_unnamed_addr global [4 x [3 x [2 x { i8, i8, i8, i8, i8, i8, i8, i8, i32 }]]], align 4

define void @func_22() {
bb:
  br label %bb13

bb13:                                             ; preds = %bb38, %bb31, %bb
  %i15 = load i32, ptr undef, align 4
  %i16 = trunc i32 %i15 to i8
  %i18 = load ptr, ptr undef, align 8
  %i19 = icmp eq ptr %i18, null
  br i1 %i19, label %bb31, label %safe_mod_func_uint8_t_u_u.exit

bb31:                                             ; preds = %bb13
  %i25 = load i8, ptr getelementptr inbounds ([4 x [3 x [2 x { i8, i8, i8, i8, i8, i8, i8, i8, i32 }]]], ptr @g_1077, i64 0, i64 2, i64 2, i64 1, i32 6), align 2
  %i28 = or i8 %i25, %i16
  store i8 %i28, ptr getelementptr inbounds ([4 x [3 x [2 x { i8, i8, i8, i8, i8, i8, i8, i8, i32 }]]], ptr @g_1077, i64 0, i64 2, i64 2, i64 1, i32 6), align 2
  %i30 = icmp ne i8 %i28, 0
  %i37.not = icmp eq i8 %i16, 0
  %or.cond = or i1 %i37.not, %i30
  br i1 %or.cond, label %bb13, label %bb38

safe_mod_func_uint8_t_u_u.exit:                   ; preds = %bb13
  br label %bb38

bb38:                                             ; preds = %safe_mod_func_uint8_t_u_u.exit, %bb31
  br label %bb13
}
