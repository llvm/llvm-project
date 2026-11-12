; RUN: llc < %s -mtriple=x86_64-unknown -mattr=sse4.1 -o /dev/null

; Testcase for PR31593.
; Revision r291120 introduced a regression and this test started failing
; because of a 'fatal error in the backend':
; Cannot select: t14: v2i64 = zero_extend_vector_inreg t18
; t18: v4i32 = bitcast t17
;   t17: v2i64,ch = load<LD16[%0](dereferenceable)> t0, FrameIndex:i64<1>, undef:i64
;     t1: i64 = FrameIndex<1>
;     t3: i64 = undef
; In function: _Z3foov
; This regression was fixed in r291535.

%struct.S = type { <2 x i64> }

declare <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32>, i32) 
define void @_Z3foov() local_unnamed_addr #2 {
entry:
  %zero = alloca %struct.S, align 16
  %e = alloca %struct.S, align 16
  %s = alloca %struct.S, align 16
  %0 = load <4 x i32>, ptr %e, align 16
  %vecext.i = extractelement <4 x i32> %0, i32 0
  %1 = call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> undef, i32 %vecext.i)
  store <4 x i32> %1, ptr %s, align 16
  ret void
}
attributes #2 = { "target-features"="+sse4.1" }
