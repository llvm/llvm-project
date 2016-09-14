; RUN: llc < %s -march=x86
; RUN: llc < %s -march=x86-64

;
; Scalars
;

define void @test_lshr_i128(i128 %x, i128 %a, i128* nocapture %r) nounwind {
entry:
	%0 = lshr i128 %x, %a
	store i128 %0, i128* %r, align 16
	ret void
}

define void @test_ashr_i128(i128 %x, i128 %a, i128* nocapture %r) nounwind {
entry:
	%0 = ashr i128 %x, %a
	store i128 %0, i128* %r, align 16
	ret void
}

define void @test_shl_i128(i128 %x, i128 %a, i128* nocapture %r) nounwind {
entry:
	%0 = shl i128 %x, %a
	store i128 %0, i128* %r, align 16
	ret void
}

define void @test_lshr_i128_outofrange(i128 %x, i128* nocapture %r) nounwind {
entry:
	%0 = lshr i128 %x, -1
	store i128 %0, i128* %r, align 16
	ret void
}

define void @test_ashr_i128_outofrange(i128 %x, i128* nocapture %r) nounwind {
entry:
	%0 = ashr i128 %x, -1
	store i128 %0, i128* %r, align 16
	ret void
}

define void @test_shl_i128_outofrange(i128 %x, i128* nocapture %r) nounwind {
entry:
	%0 = shl i128 %x, -1
	store i128 %0, i128* %r, align 16
	ret void
}

;
; Vectors
;

define void @test_lshr_v2i128(<2 x i128> %x, <2 x i128> %a, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = lshr <2 x i128> %x, %a
	store <2 x i128> %0, <2 x i128>* %r, align 16
	ret void
}

define void @test_ashr_v2i128(<2 x i128> %x, <2 x i128> %a, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = ashr <2 x i128> %x, %a
	store <2 x i128> %0, <2 x i128>* %r, align 16
	ret void
}

define void @test_shl_v2i128(<2 x i128> %x, <2 x i128> %a, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = shl <2 x i128> %x, %a
	store <2 x i128> %0, <2 x i128>* %r, align 16
	ret void
}

define void @test_lshr_v2i128_outofrange(<2 x i128> %x, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = lshr <2 x i128> %x, <i128 -1, i128 -1>
	store <2 x i128> %0, <2 x i128>* %r, align 16
	ret void
}

define void @test_ashr_v2i128_outofrange(<2 x i128> %x, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = ashr <2 x i128> %x, <i128 -1, i128 -1>
	store <2 x i128> %0, <2 x i128>* %r, align 16
	ret void
}

define void @test_shl_v2i128_outofrange(<2 x i128> %x, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = shl <2 x i128> %x, <i128 -1, i128 -1>
	store <2 x i128> %0, <2 x i128>* %r, align 16
	ret void
}

define void @test_lshr_v2i128_outofrange_sum(<2 x i128> %x, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = lshr <2 x i128> %x, <i128 -1, i128 -1>
	%1 = lshr <2 x i128> %0, <i128  1, i128  1>
	store <2 x i128> %1, <2 x i128>* %r, align 16
	ret void
}

define void @test_ashr_v2i128_outofrange_sum(<2 x i128> %x, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = ashr <2 x i128> %x, <i128 -1, i128 -1>
	%1 = ashr <2 x i128> %0, <i128  1, i128  1>
	store <2 x i128> %1, <2 x i128>* %r, align 16
	ret void
}

define void @test_shl_v2i128_outofrange_sum(<2 x i128> %x, <2 x i128>* nocapture %r) nounwind {
entry:
	%0 = shl <2 x i128> %x, <i128 -1, i128 -1>
	%1 = shl <2 x i128> %0, <i128  1, i128  1>
	store <2 x i128> %1, <2 x i128>* %r, align 16
	ret void
}

;
; Combines
;

define <2 x i256> @shl_sext_shl_outofrange(<2 x i128> %a0) {
  %1 = shl <2 x i128> %a0, <i128 -1, i128 -1>
  %2 = sext <2 x i128> %1 to <2 x i256>
  %3 = shl <2 x i256> %2, <i256 128, i256 128>
  ret <2 x i256> %3
}

define <2 x i256> @shl_zext_shl_outofrange(<2 x i128> %a0) {
  %1 = shl <2 x i128> %a0, <i128 -1, i128 -1>
  %2 = zext <2 x i128> %1 to <2 x i256>
  %3 = shl <2 x i256> %2, <i256 128, i256 128>
  ret <2 x i256> %3
}

define <2 x i256> @shl_zext_lshr_outofrange(<2 x i128> %a0) {
  %1 = lshr <2 x i128> %a0, <i128 -1, i128 -1>
  %2 = zext <2 x i128> %1 to <2 x i256>
  %3 = shl <2 x i256> %2, <i256 128, i256 128>
  ret <2 x i256> %3
}

define i128 @lshr_shl_mask(i128 %a0) {
  %1 = shl i128 %a0, 1
  %2 = lshr i128 %1, 1
  ret i128 %2
}
