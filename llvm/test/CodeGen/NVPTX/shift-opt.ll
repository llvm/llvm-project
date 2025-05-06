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

define <2 x i16> @test_vec(<2 x i16> %x, <2 x i8> %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
; x, y - vectors
;
; CHECK-LABEL: test_vec
; CHECK: ld.param.u32 %[[X:r[0-9]+]], [test_vec_param_0];
; CHECK: ld.param.u32 %[[P1:r[0-9]+]], [test_vec_param_1];
; CHECK: and.b32      %[[Y:r[0-9]+]], %[[P1]], 16711935;
; CHECK: mov.b32      {%[[X1:rs[0-9]+]], %[[X2:rs[0-9]+]]}, %[[X]];
; CHECK: shr.u16      %[[SHR2:rs[0-9]+]], %[[X2]], 5;
; CHECK: shr.u16      %[[SHR1:rs[0-9]+]], %[[X1]], 5;
; CHECK: mov.b32      %[[SHR:r[0-9]+]], {%[[SHR1]], %[[SHR2]]};
; CHECK: or.b32       %[[LOP:r[0-9]+]], %[[SHR]], %[[Y]];
; CHECK: st.param.b32 [func_retval0], %[[LOP]];
;
  %ext = zext <2 x i8> %y to <2 x i16>
  %shl = shl <2 x i16> %ext, splat(i16 5)
  %or = or <2 x i16> %x, %shl
  %srl = lshr <2 x i16> %or, splat(i16 5)
  ret <2 x i16> %srl
}

define i64 @test_negative_c(i64 %x, i32 %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 > leadingzeros(zext(y)).
;
; CHECK-LABEL: test_negative_c
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test_negative_c_param_0];
; CHECK: ld.param.u32 %[[Y:rd[0-9]+]], [test_negative_c_param_1];
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

declare void @use(i64)

define i64 @test_negative_use_lop(i64 %x, i32 %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
; multiple usage of "or"
;
; CHECK-LABEL: test_negative_use_lop
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test_negative_use_lop_param_0];
; CHECK: ld.param.u32 %[[Y:r[0-9]+]], [test_negative_use_lop_param_1];
; CHECK: mul.wide.u32 %[[SHL:rd[0-9]+]], %[[Y]], 32;
; CHECK: or.b64       %[[OR:rd[0-9]+]], %[[X]], %[[SHL]];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[OR]], 5;
; CHECK: { // callseq
; CHECK:      st.param.b64    [param0], %[[OR]];
; CHECK: } // callseq
; CHECK: st.param.b64 [func_retval0], %[[SHR]];
;
  %ext = zext i32 %y to i64
  %shl = shl i64 %ext, 5
  %or = or i64 %x, %shl
  %srl = lshr i64 %or, 5
  call void @use(i64 %or)
  ret i64 %srl
}


define i64 @test_negative_use_shl(i64 %x, i32 %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
; multiple usage of "shl"
;
; CHECK-LABEL: test_negative_use_shl
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test_negative_use_shl_param_0];
; CHECK: ld.param.u32 %[[Y:r[0-9]+]], [test_negative_use_shl_param_1];
; CHECK: mul.wide.u32 %[[SHL:rd[0-9]+]], %[[Y]], 32;
; CHECK: or.b64       %[[OR:rd[0-9]+]], %[[X]], %[[SHL]];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[OR]], 5;
; CHECK: { // callseq
; CHECK:      st.param.b64    [param0], %[[SHL]];
; CHECK: } // callseq
; CHECK: st.param.b64 [func_retval0], %[[SHR]];
;
  %ext = zext i32 %y to i64
  %shl = shl i64 %ext, 5
  %or = or i64 %x, %shl
  %srl = lshr i64 %or, 5
  call void @use(i64 %shl)
  ret i64 %srl
}
