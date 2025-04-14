; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s

define i64 @test1(i64 %x, i32 %y) {
;
; srl (or (x, shl(zext(y),c1)),c1) -> or(srl(x,c1), zext(y))
; c1 <= leadingzeros(zext(y))
;
; CHECK-LABEL: test1
; CHECK: ld.param.u64 %[[X:rd[0-9]+]], [test1_param_0];
; CHECK: ld.param.u32 %[[Y:rd[0-9]+]], [test1_param_1];
; CHECK: shr.u64      %[[SHR:rd[0-9]+]], %[[X]], 5;
; CHECK: or.b64       %[[OR:rd[0-9]+]], %[[SHR]], %[[Y]];
; CHECK: st.param.b64 [func_retval0], %[[OR]];
;
  %ext = zext i32 %y to i64
  %shl = shl i64 %ext, 5
  %or = or i64 %x, %shl
  %srl = lshr i64 %or, 5
  ret i64 %srl
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
