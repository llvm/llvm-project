; RUN: llc -mtriple=arm-eabi -mattr=+neon -O0 -optimize-regalloc -regalloc=basic %s -o /dev/null

; This test would crash the rewriter when trying to handle a spill after one of
; the @llvm.arm.neon.vld3.v8i8.p0 defined three parts of a register.

%struct.__neon_int8x8x3_t = type { <8 x i8>, <8 x i8>, <8 x i8> }

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr, i32) nounwind readonly

declare void @llvm.arm.neon.vst3.p0.v8i8(ptr, <8 x i8>, <8 x i8>, <8 x i8>, i32) nounwind

define <8 x i8> @t3(ptr %A1, ptr %A2, ptr %A3, ptr %A4, ptr %A5, ptr %A6, ptr %A7, ptr %A8, ptr %B) nounwind {
  %tmp1b = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr %A2, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=2]
  %tmp2b = extractvalue %struct.__neon_int8x8x3_t %tmp1b, 0 ; <<8 x i8>> [#uses=1]
  %tmp4b = extractvalue %struct.__neon_int8x8x3_t %tmp1b, 1 ; <<8 x i8>> [#uses=1]
  %tmp1d = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr %A4, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=2]
  %tmp2d = extractvalue %struct.__neon_int8x8x3_t %tmp1d, 0 ; <<8 x i8>> [#uses=1]
  %tmp4d = extractvalue %struct.__neon_int8x8x3_t %tmp1d, 1 ; <<8 x i8>> [#uses=1]
  %tmp1e = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr %A5, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=1]
  %tmp2e = extractvalue %struct.__neon_int8x8x3_t %tmp1e, 0 ; <<8 x i8>> [#uses=1]
  %tmp1f = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr %A6, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=1]
  %tmp2f = extractvalue %struct.__neon_int8x8x3_t %tmp1f, 0 ; <<8 x i8>> [#uses=1]
  %tmp1g = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr %A7, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=2]
  %tmp2g = extractvalue %struct.__neon_int8x8x3_t %tmp1g, 0 ; <<8 x i8>> [#uses=1]
  %tmp4g = extractvalue %struct.__neon_int8x8x3_t %tmp1g, 1 ; <<8 x i8>> [#uses=1]
  %tmp1h = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0(ptr %A8, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=2]
  %tmp2h = extractvalue %struct.__neon_int8x8x3_t %tmp1h, 0 ; <<8 x i8>> [#uses=1]
  %tmp3h = extractvalue %struct.__neon_int8x8x3_t %tmp1h, 2 ; <<8 x i8>> [#uses=1]
  %tmp2bd = add <8 x i8> %tmp2b, %tmp2d           ; <<8 x i8>> [#uses=1]
  %tmp4bd = add <8 x i8> %tmp4b, %tmp4d           ; <<8 x i8>> [#uses=1]
  %tmp2abcd = mul <8 x i8> undef, %tmp2bd         ; <<8 x i8>> [#uses=1]
  %tmp4abcd = mul <8 x i8> undef, %tmp4bd         ; <<8 x i8>> [#uses=2]
  call void @llvm.arm.neon.vst3.p0.v8i8(ptr %A1, <8 x i8> %tmp4abcd, <8 x i8> zeroinitializer, <8 x i8> %tmp2abcd, i32 1)
  %tmp2ef = sub <8 x i8> %tmp2e, %tmp2f           ; <<8 x i8>> [#uses=1]
  %tmp2gh = sub <8 x i8> %tmp2g, %tmp2h           ; <<8 x i8>> [#uses=1]
  %tmp3gh = sub <8 x i8> zeroinitializer, %tmp3h  ; <<8 x i8>> [#uses=1]
  %tmp4ef = sub <8 x i8> zeroinitializer, %tmp4g  ; <<8 x i8>> [#uses=1]
  %tmp2efgh = mul <8 x i8> %tmp2ef, %tmp2gh       ; <<8 x i8>> [#uses=1]
  %tmp3efgh = mul <8 x i8> undef, %tmp3gh         ; <<8 x i8>> [#uses=1]
  %tmp4efgh = mul <8 x i8> %tmp4ef, undef         ; <<8 x i8>> [#uses=2]
  call void @llvm.arm.neon.vst3.p0.v8i8(ptr %A2, <8 x i8> %tmp4efgh, <8 x i8> %tmp3efgh, <8 x i8> %tmp2efgh, i32 1)
  %tmp4 = sub <8 x i8> %tmp4efgh, %tmp4abcd       ; <<8 x i8>> [#uses=1]
  tail call void @llvm.arm.neon.vst3.p0.v8i8(ptr %B, <8 x i8> zeroinitializer, <8 x i8> undef, <8 x i8> undef, i32 1)
  ret <8 x i8> %tmp4
}
