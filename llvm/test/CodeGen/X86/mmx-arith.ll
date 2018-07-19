; RUN: llc < %s -mtriple=i686-- -mattr=+mmx,+sse2 | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -mtriple=x86_64-- -mattr=+mmx,+sse2 | FileCheck -check-prefix=X64 %s

;; A basic sanity check to make sure that MMX arithmetic actually compiles.
;; First is a straight translation of the original with bitcasts as needed.

; X32-LABEL: test0
; X64-LABEL: test0
define void @test0(x86_mmx* %A, x86_mmx* %B) {
entry:
  %tmp1 = load x86_mmx, x86_mmx* %A
  %tmp3 = load x86_mmx, x86_mmx* %B
  %tmp1a = bitcast x86_mmx %tmp1 to <8 x i8>
  %tmp3a = bitcast x86_mmx %tmp3 to <8 x i8>
  %tmp4 = add <8 x i8> %tmp1a, %tmp3a
  %tmp4a = bitcast <8 x i8> %tmp4 to x86_mmx
  store x86_mmx %tmp4a, x86_mmx* %A
  %tmp7 = load x86_mmx, x86_mmx* %B
  %tmp12 = tail call x86_mmx @llvm.x86.mmx.padds.b(x86_mmx %tmp4a, x86_mmx %tmp7)
  store x86_mmx %tmp12, x86_mmx* %A
  %tmp16 = load x86_mmx, x86_mmx* %B
  %tmp21 = tail call x86_mmx @llvm.x86.mmx.paddus.b(x86_mmx %tmp12, x86_mmx %tmp16)
  store x86_mmx %tmp21, x86_mmx* %A
  %tmp27 = load x86_mmx, x86_mmx* %B
  %tmp21a = bitcast x86_mmx %tmp21 to <8 x i8>
  %tmp27a = bitcast x86_mmx %tmp27 to <8 x i8>
  %tmp28 = sub <8 x i8> %tmp21a, %tmp27a
  %tmp28a = bitcast <8 x i8> %tmp28 to x86_mmx
  store x86_mmx %tmp28a, x86_mmx* %A
  %tmp31 = load x86_mmx, x86_mmx* %B
  %tmp36 = tail call x86_mmx @llvm.x86.mmx.psubs.b(x86_mmx %tmp28a, x86_mmx %tmp31)
  store x86_mmx %tmp36, x86_mmx* %A
  %tmp40 = load x86_mmx, x86_mmx* %B
  %tmp45 = tail call x86_mmx @llvm.x86.mmx.psubus.b(x86_mmx %tmp36, x86_mmx %tmp40)
  store x86_mmx %tmp45, x86_mmx* %A
  %tmp51 = load x86_mmx, x86_mmx* %B
  %tmp45a = bitcast x86_mmx %tmp45 to <8 x i8>
  %tmp51a = bitcast x86_mmx %tmp51 to <8 x i8>
  %tmp52 = mul <8 x i8> %tmp45a, %tmp51a
  %tmp52a = bitcast <8 x i8> %tmp52 to x86_mmx
  store x86_mmx %tmp52a, x86_mmx* %A
  %tmp57 = load x86_mmx, x86_mmx* %B
  %tmp57a = bitcast x86_mmx %tmp57 to <8 x i8>
  %tmp58 = and <8 x i8> %tmp52, %tmp57a
  %tmp58a = bitcast <8 x i8> %tmp58 to x86_mmx
  store x86_mmx %tmp58a, x86_mmx* %A
  %tmp63 = load x86_mmx, x86_mmx* %B
  %tmp63a = bitcast x86_mmx %tmp63 to <8 x i8>
  %tmp64 = or <8 x i8> %tmp58, %tmp63a
  %tmp64a = bitcast <8 x i8> %tmp64 to x86_mmx
  store x86_mmx %tmp64a, x86_mmx* %A
  %tmp69 = load x86_mmx, x86_mmx* %B
  %tmp69a = bitcast x86_mmx %tmp69 to <8 x i8>
  %tmp64b = bitcast x86_mmx %tmp64a to <8 x i8>
  %tmp70 = xor <8 x i8> %tmp64b, %tmp69a
  %tmp70a = bitcast <8 x i8> %tmp70 to x86_mmx
  store x86_mmx %tmp70a, x86_mmx* %A
  tail call void @llvm.x86.mmx.emms()
  ret void
}

; X32-LABEL: test1
; X64-LABEL: test1
define void @test1(x86_mmx* %A, x86_mmx* %B) {
entry:
  %tmp1 = load x86_mmx, x86_mmx* %A
  %tmp3 = load x86_mmx, x86_mmx* %B
  %tmp1a = bitcast x86_mmx %tmp1 to <2 x i32>
  %tmp3a = bitcast x86_mmx %tmp3 to <2 x i32>
  %tmp4 = add <2 x i32> %tmp1a, %tmp3a
  %tmp4a = bitcast <2 x i32> %tmp4 to x86_mmx
  store x86_mmx %tmp4a, x86_mmx* %A
  %tmp9 = load x86_mmx, x86_mmx* %B
  %tmp9a = bitcast x86_mmx %tmp9 to <2 x i32>
  %tmp10 = sub <2 x i32> %tmp4, %tmp9a
  %tmp10a = bitcast <2 x i32> %tmp4 to x86_mmx
  store x86_mmx %tmp10a, x86_mmx* %A
  %tmp15 = load x86_mmx, x86_mmx* %B
  %tmp10b = bitcast x86_mmx %tmp10a to <2 x i32>
  %tmp15a = bitcast x86_mmx %tmp15 to <2 x i32>
  %tmp16 = mul <2 x i32> %tmp10b, %tmp15a
  %tmp16a = bitcast <2 x i32> %tmp16 to x86_mmx
  store x86_mmx %tmp16a, x86_mmx* %A
  %tmp21 = load x86_mmx, x86_mmx* %B
  %tmp16b = bitcast x86_mmx %tmp16a to <2 x i32>
  %tmp21a = bitcast x86_mmx %tmp21 to <2 x i32>
  %tmp22 = and <2 x i32> %tmp16b, %tmp21a
  %tmp22a = bitcast <2 x i32> %tmp22 to x86_mmx
  store x86_mmx %tmp22a, x86_mmx* %A
  %tmp27 = load x86_mmx, x86_mmx* %B
  %tmp22b = bitcast x86_mmx %tmp22a to <2 x i32>
  %tmp27a = bitcast x86_mmx %tmp27 to <2 x i32>
  %tmp28 = or <2 x i32> %tmp22b, %tmp27a
  %tmp28a = bitcast <2 x i32> %tmp28 to x86_mmx
  store x86_mmx %tmp28a, x86_mmx* %A
  %tmp33 = load x86_mmx, x86_mmx* %B
  %tmp28b = bitcast x86_mmx %tmp28a to <2 x i32>
  %tmp33a = bitcast x86_mmx %tmp33 to <2 x i32>
  %tmp34 = xor <2 x i32> %tmp28b, %tmp33a
  %tmp34a = bitcast <2 x i32> %tmp34 to x86_mmx
  store x86_mmx %tmp34a, x86_mmx* %A
  tail call void @llvm.x86.mmx.emms( )
  ret void
}

; X32-LABEL: test2
; X64-LABEL: test2
define void @test2(x86_mmx* %A, x86_mmx* %B) {
entry:
  %tmp1 = load x86_mmx, x86_mmx* %A
  %tmp3 = load x86_mmx, x86_mmx* %B
  %tmp1a = bitcast x86_mmx %tmp1 to <4 x i16>
  %tmp3a = bitcast x86_mmx %tmp3 to <4 x i16>
  %tmp4 = add <4 x i16> %tmp1a, %tmp3a
  %tmp4a = bitcast <4 x i16> %tmp4 to x86_mmx
  store x86_mmx %tmp4a, x86_mmx* %A
  %tmp7 = load x86_mmx, x86_mmx* %B
  %tmp12 = tail call x86_mmx @llvm.x86.mmx.padds.w(x86_mmx %tmp4a, x86_mmx %tmp7)
  store x86_mmx %tmp12, x86_mmx* %A
  %tmp16 = load x86_mmx, x86_mmx* %B
  %tmp21 = tail call x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx %tmp12, x86_mmx %tmp16)
  store x86_mmx %tmp21, x86_mmx* %A
  %tmp27 = load x86_mmx, x86_mmx* %B
  %tmp21a = bitcast x86_mmx %tmp21 to <4 x i16>
  %tmp27a = bitcast x86_mmx %tmp27 to <4 x i16>
  %tmp28 = sub <4 x i16> %tmp21a, %tmp27a
  %tmp28a = bitcast <4 x i16> %tmp28 to x86_mmx
  store x86_mmx %tmp28a, x86_mmx* %A
  %tmp31 = load x86_mmx, x86_mmx* %B
  %tmp36 = tail call x86_mmx @llvm.x86.mmx.psubs.w(x86_mmx %tmp28a, x86_mmx %tmp31)
  store x86_mmx %tmp36, x86_mmx* %A
  %tmp40 = load x86_mmx, x86_mmx* %B
  %tmp45 = tail call x86_mmx @llvm.x86.mmx.psubus.w(x86_mmx %tmp36, x86_mmx %tmp40)
  store x86_mmx %tmp45, x86_mmx* %A
  %tmp51 = load x86_mmx, x86_mmx* %B
  %tmp45a = bitcast x86_mmx %tmp45 to <4 x i16>
  %tmp51a = bitcast x86_mmx %tmp51 to <4 x i16>
  %tmp52 = mul <4 x i16> %tmp45a, %tmp51a
  %tmp52a = bitcast <4 x i16> %tmp52 to x86_mmx
  store x86_mmx %tmp52a, x86_mmx* %A
  %tmp55 = load x86_mmx, x86_mmx* %B
  %tmp60 = tail call x86_mmx @llvm.x86.mmx.pmulh.w(x86_mmx %tmp52a, x86_mmx %tmp55)
  store x86_mmx %tmp60, x86_mmx* %A
  %tmp64 = load x86_mmx, x86_mmx* %B
  %tmp69 = tail call x86_mmx @llvm.x86.mmx.pmadd.wd(x86_mmx %tmp60, x86_mmx %tmp64)
  %tmp70 = bitcast x86_mmx %tmp69 to x86_mmx
  store x86_mmx %tmp70, x86_mmx* %A
  %tmp75 = load x86_mmx, x86_mmx* %B
  %tmp70a = bitcast x86_mmx %tmp70 to <4 x i16>
  %tmp75a = bitcast x86_mmx %tmp75 to <4 x i16>
  %tmp76 = and <4 x i16> %tmp70a, %tmp75a
  %tmp76a = bitcast <4 x i16> %tmp76 to x86_mmx
  store x86_mmx %tmp76a, x86_mmx* %A
  %tmp81 = load x86_mmx, x86_mmx* %B
  %tmp76b = bitcast x86_mmx %tmp76a to <4 x i16>
  %tmp81a = bitcast x86_mmx %tmp81 to <4 x i16>
  %tmp82 = or <4 x i16> %tmp76b, %tmp81a
  %tmp82a = bitcast <4 x i16> %tmp82 to x86_mmx
  store x86_mmx %tmp82a, x86_mmx* %A
  %tmp87 = load x86_mmx, x86_mmx* %B
  %tmp82b = bitcast x86_mmx %tmp82a to <4 x i16>
  %tmp87a = bitcast x86_mmx %tmp87 to <4 x i16>
  %tmp88 = xor <4 x i16> %tmp82b, %tmp87a
  %tmp88a = bitcast <4 x i16> %tmp88 to x86_mmx
  store x86_mmx %tmp88a, x86_mmx* %A
  tail call void @llvm.x86.mmx.emms( )
  ret void
}

; X32-LABEL: test3
define <1 x i64> @test3(<1 x i64>* %a, <1 x i64>* %b, i32 %count) nounwind {
entry:
  %tmp2942 = icmp eq i32 %count, 0
  br i1 %tmp2942, label %bb31, label %bb26

bb26:
; X32:  addl
; X32:  adcl
  %i.037.0 = phi i32 [ 0, %entry ], [ %tmp25, %bb26 ]
  %sum.035.0 = phi <1 x i64> [ zeroinitializer, %entry ], [ %tmp22, %bb26 ]
  %tmp13 = getelementptr <1 x i64>, <1 x i64>* %b, i32 %i.037.0
  %tmp14 = load <1 x i64>, <1 x i64>* %tmp13
  %tmp18 = getelementptr <1 x i64>, <1 x i64>* %a, i32 %i.037.0
  %tmp19 = load <1 x i64>, <1 x i64>* %tmp18
  %tmp21 = add <1 x i64> %tmp19, %tmp14
  %tmp22 = add <1 x i64> %tmp21, %sum.035.0
  %tmp25 = add i32 %i.037.0, 1
  %tmp29 = icmp ult i32 %tmp25, %count
  br i1 %tmp29, label %bb26, label %bb31

bb31:
  %sum.035.1 = phi <1 x i64> [ zeroinitializer, %entry ], [ %tmp22, %bb26 ]
  ret <1 x i64> %sum.035.1
}

; There are no MMX operations here, so we use XMM or i64.
; X64-LABEL: ti8
define void @ti8(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to <8 x i8>
  %tmp2 = bitcast double %b to <8 x i8>
  %tmp3 = add <8 x i8> %tmp1, %tmp2
; X64:  paddb
  store <8 x i8> %tmp3, <8 x i8>* null
  ret void
}

; X64-LABEL: ti16
define void @ti16(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to <4 x i16>
  %tmp2 = bitcast double %b to <4 x i16>
  %tmp3 = add <4 x i16> %tmp1, %tmp2
; X64:  paddw
  store <4 x i16> %tmp3, <4 x i16>* null
  ret void
}

; X64-LABEL: ti32
define void @ti32(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to <2 x i32>
  %tmp2 = bitcast double %b to <2 x i32>
  %tmp3 = add <2 x i32> %tmp1, %tmp2
; X64:  paddd
  store <2 x i32> %tmp3, <2 x i32>* null
  ret void
}

; X64-LABEL: ti64
define void @ti64(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to <1 x i64>
  %tmp2 = bitcast double %b to <1 x i64>
  %tmp3 = add <1 x i64> %tmp1, %tmp2
; X64:  addq
  store <1 x i64> %tmp3, <1 x i64>* null
  ret void
}

; MMX intrinsics calls get us MMX instructions.
; X64-LABEL: ti8a
define void @ti8a(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to x86_mmx
; X64: movdq2q
  %tmp2 = bitcast double %b to x86_mmx
; X64: movdq2q
  %tmp3 = tail call x86_mmx @llvm.x86.mmx.padd.b(x86_mmx %tmp1, x86_mmx %tmp2)
  store x86_mmx %tmp3, x86_mmx* null
  ret void
}

; X64-LABEL: ti16a
define void @ti16a(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to x86_mmx
; X64: movdq2q
  %tmp2 = bitcast double %b to x86_mmx
; X64: movdq2q
  %tmp3 = tail call x86_mmx @llvm.x86.mmx.padd.w(x86_mmx %tmp1, x86_mmx %tmp2)
  store x86_mmx %tmp3, x86_mmx* null
  ret void
}

; X64-LABEL: ti32a
define void @ti32a(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to x86_mmx
; X64: movdq2q
  %tmp2 = bitcast double %b to x86_mmx
; X64: movdq2q
  %tmp3 = tail call x86_mmx @llvm.x86.mmx.padd.d(x86_mmx %tmp1, x86_mmx %tmp2)
  store x86_mmx %tmp3, x86_mmx* null
  ret void
}

; X64-LABEL: ti64a
define void @ti64a(double %a, double %b) nounwind {
entry:
  %tmp1 = bitcast double %a to x86_mmx
; X64: movdq2q
  %tmp2 = bitcast double %b to x86_mmx
; X64: movdq2q
  %tmp3 = tail call x86_mmx @llvm.x86.mmx.padd.q(x86_mmx %tmp1, x86_mmx %tmp2)
  store x86_mmx %tmp3, x86_mmx* null
  ret void
}

declare x86_mmx @llvm.x86.mmx.padd.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.d(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.q(x86_mmx, x86_mmx)

declare x86_mmx @llvm.x86.mmx.paddus.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psubus.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psubus.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.pmulh.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.pmadd.wd(x86_mmx, x86_mmx)

declare void @llvm.x86.mmx.emms()

declare x86_mmx @llvm.x86.mmx.padds.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padds.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psubs.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psubs.w(x86_mmx, x86_mmx)

