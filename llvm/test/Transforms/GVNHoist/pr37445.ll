; RUN: opt < %s -passes='early-cse<memssa>,gvn-hoist' -earlycse-debug-hash -S | FileCheck %s

; Make sure opt won't crash and that this pair of
; instructions (load, icmp) are not hoisted.
; Although it is safe to hoist the loads from bb45 to
; bb41, gvn-hoist does not have appropriate mechanism
; to handle corner cases (see PR46874) when these instructions
; were hoisted.
; FIXME: Hoist loads from bb58 and bb45 to bb41.

@g_10 = external global i32, align 4
@g_536 = external global ptr, align 8
@g_1629 = external global ptr, align 8
@g_963 = external global ptr, align 8
@g_1276 = external global ptr, align 8

;CHECK-LABEL: @func_22

define void @func_22(ptr %arg, ptr %arg1) {
bb:
  br label %bb12

bb12:
  %tmp3.0 = phi i32 [ undef, %bb ], [ %tmp40, %bb36 ]
  %tmp7.0 = phi i32 [ undef, %bb ], [ %spec.select, %bb36 ]
  %tmp14 = icmp eq i32 %tmp3.0, 6
  br i1 %tmp14, label %bb41, label %bb15

bb15:
  %tmp183 = trunc i16 0 to i8
  %tmp20 = load ptr, ptr @g_536, align 8
  %tmp21 = load i8, ptr %tmp20, align 1
  %tmp23 = or i8 %tmp21, %tmp183
  store i8 %tmp23, ptr %tmp20, align 1
  %tmp5.i = icmp eq i8 %tmp23, 0
  br i1 %tmp5.i, label %safe_div_func_uint8_t_u_u.exit, label %bb8.i

bb8.i:
  %0 = udiv i8 1, %tmp23
  br label %safe_div_func_uint8_t_u_u.exit

safe_div_func_uint8_t_u_u.exit:
  %tmp13.in.i = phi i8 [ %0, %bb8.i ], [ 1, %bb15 ]
  %tmp31 = icmp eq i8 %tmp13.in.i, 0
  %spec.select = select i1 %tmp31, i32 %tmp7.0, i32 53
  %tmp35 = icmp eq i32 %spec.select, 0
  br i1 %tmp35, label %bb36, label %bb41

bb36:
  %tmp38 = sext i32 %tmp3.0 to i64
  %tmp40 = trunc i64 %tmp38 to i32
  br label %bb12

;CHECK: bb41:

bb41:
  %tmp43 = load i32, ptr %arg, align 4
  %tmp44 = icmp eq i32 %tmp43, 0
  br i1 %tmp44, label %bb52, label %bb45

;CHECK:     bb45:
;CHECK:   %tmp47 = load i32, ptr %arg1, align 4
;CHECK:   %tmp48 = icmp eq i32 %tmp47, 0

bb45:
  %tmp47 = load i32, ptr %arg1, align 4
  %tmp48 = icmp eq i32 %tmp47, 0
  br i1 %tmp48, label %bb50, label %bb64

bb50:
  %tmp51 = load volatile ptr, ptr @g_963, align 8
  unreachable

bb52:
  %tmp8.0 = phi i32 [ undef, %bb41 ], [ %tmp57, %bb55 ]
  %tmp54 = icmp slt i32 %tmp8.0, 3
  br i1 %tmp54, label %bb55, label %bb58

bb55:
  %tmp57 = add nsw i32 %tmp8.0, 1
  br label %bb52

;CHECK: bb58:
;CHECK: %tmp60 = load i32, ptr %arg1, align 4
;CHECK: %tmp61 = icmp eq i32 %tmp60, 0
;CHECK: bb62:
;CHECK: load
;CHECK: bb64:
;CHECK: load

bb58:
  %tmp60 = load i32, ptr %arg1, align 4
  %tmp61 = icmp eq i32 %tmp60, 0
  br i1 %tmp61, label %bb62, label %bb64

bb62:
  %tmp63 = load volatile ptr, ptr @g_1276, align 8
  unreachable

bb64:
  %tmp65 = load volatile ptr, ptr @g_1629, align 8
  unreachable

; uselistorder directives
  uselistorder i32 %spec.select, { 1, 0 }
  uselistorder ptr %arg1, { 1, 0 }
  uselistorder label %bb64, { 1, 0 }
  uselistorder label %bb52, { 1, 0 }
  uselistorder label %bb41, { 1, 0 }
  uselistorder label %safe_div_func_uint8_t_u_u.exit, { 1, 0 }
}

define zeroext i8 @safe_div_func_uint8_t_u_u(i8 zeroext %arg, i8 zeroext %arg1) {
bb:
  %tmp5 = icmp eq i8 %arg1, 0
  br i1 %tmp5, label %bb12, label %bb8

bb8:
  %0 = udiv i8 %arg, %arg1
  br label %bb12

bb12:
  %tmp13.in = phi i8 [ %0, %bb8 ], [ %arg, %bb ]
  ret i8 %tmp13.in
}
