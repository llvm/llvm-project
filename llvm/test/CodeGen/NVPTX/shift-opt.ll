; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s

define i64 @test_or(i64 %x, i32 %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
;
; CHECK-LABEL: test_or
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test_or_param_0];
; CHECK: ld.param.u32 %[[Y:rd[0-9]+]], [test_or_param_1];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[X]], 5;
; CHECK: or.b64       %[[LOP:rd[0-9]+]], %[[SHR]], %[[Y]];
; CHECK: st.param.b64 [func_retval0], %[[LOP]];
;
  %ext = zext i32 %y to i64
  %shl = shl i64 %ext, 5
  %or = or i64 %x, %shl
  %srl = lshr i64 %or, 5
  ret i64 %srl
}

define i64 @test_xor(i64 %x, i32 %y) {
;
; srl (xor (x, shl(zext(y),c1)),c1) -> xor(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
;
; CHECK-LABEL: test_xor
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test_xor_param_0];
; CHECK: ld.param.u32 %[[Y:rd[0-9]+]], [test_xor_param_1];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[X]], 5;
; CHECK: xor.b64      %[[LOP:rd[0-9]+]], %[[SHR]], %[[Y]];
; CHECK: st.param.b64 [func_retval0], %[[LOP]];
;
  %ext = zext i32 %y to i64
  %shl = shl i64 %ext, 5
  %or = xor i64 %x, %shl
  %srl = lshr i64 %or, 5
  ret i64 %srl
}

define i64 @test_and(i64 %x, i32 %y) {
;
; srl (and (x, shl(zext(y),c1)),c1) -> and(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
;
; CHECK-LABEL: test_and
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test_and_param_0];
; CHECK: ld.param.u32 %[[Y:rd[0-9]+]], [test_and_param_1];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[X]], 5;
; CHECK: and.b64      %[[LOP:rd[0-9]+]], %[[SHR]], %[[Y]];
; CHECK: st.param.b64 [func_retval0], %[[LOP]];
;
  %ext = zext i32 %y to i64
  %shl = shl i64 %ext, 5
  %or = and i64 %x, %shl
  %srl = lshr i64 %or, 5
  ret i64 %srl
}

define <2 x i64> @test_or_vec(<2 x i64> %x, <2 x i32> %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
;
; CHECK-LABEL: test_or
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test_or_param_0];
; CHECK: ld.param.u32 %[[Y:rd[0-9]+]], [test_or_param_1];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[X]], 5;
; CHECK: or.b64       %[[LOP:rd[0-9]+]], %[[SHR]], %[[Y]];
; CHECK: st.param.b64 [func_retval0], %[[LOP]];
;
  %ext = zext <2 x i32> %y to <2 x i64>
  %shl = shl <2 x i64> %ext, splat(i64 5)
  %or = or <2 x i64> %x, %shl
  %srl = lshr <2 x i64> %or, splat(i64 5)
  ret <2 x i64> %srl
}

define i64 @test2(i64 %x, i32 %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 > leadingzeros(zext(y)).
;
; CHECK-LABEL: test2
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test2_param_0];
; CHECK: ld.param.u32 %[[Y:rd[0-9]+]], [test2_param_1];
; CHECK: shl.b64      %[[SHL:rd[0-9]+]], %[[Y]], 33;
; CHECK: or.b64       %[[OR:rd[0-9]+]], %[[X]], %[[SHL]];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[OR]], 33;
; CHECK: st.param.b64 [func_retval0], %[[SHR]];
;
  %ext = zext i32 %y to i64
  %shl = shl i64 %ext, 33
  %or = or i64 %x, %shl
  %srl = lshr i64 %or, 33
  ret i64 %srl
}
