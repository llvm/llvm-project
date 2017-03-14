; RUN: llc -mtriple=x86_64-unknown-unknown -force-split-store < %s | FileCheck %s

; CHECK-LABEL: int32_float_pair
; CHECK-DAG: movl %edi, (%rsi)
; CHECK-DAG: movss %xmm0, 4(%rsi)
define void @int32_float_pair(i32 %tmp1, float %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i32 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: float_int32_pair
; CHECK-DAG: movss %xmm0, (%rsi)
; CHECK-DAG: movl %edi, 4(%rsi)
define void @float_int32_pair(float %tmp1, i32 %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp1 to i32
  %t1 = zext i32 %tmp2 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i32 %t0 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int16_float_pair
; CHECK-DAG: movzwl	%di, %eax
; CHECK-DAG: movl %eax, (%rsi)
; CHECK-DAG: movss %xmm0, 4(%rsi)
define void @int16_float_pair(i16 signext %tmp1, float %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i16 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int8_float_pair
; CHECK-DAG: movzbl	%dil, %eax
; CHECK-DAG: movl %eax, (%rsi)
; CHECK-DAG: movss %xmm0, 4(%rsi)
define void @int8_float_pair(i8 signext %tmp1, float %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i8 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int32_int32_pair
; CHECK: movl	%edi, (%rdx)
; CHECK: movl	%esi, 4(%rdx)
define void @int32_int32_pair(i32 %tmp1, i32 %tmp2, i64* %ref.tmp) {
entry:
  %t1 = zext i32 %tmp2 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i32 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int16_int16_pair
; CHECK: movw	%di, (%rdx)
; CHECK: movw	%si, 2(%rdx)
define void @int16_int16_pair(i16 signext %tmp1, i16 signext %tmp2, i32* %ref.tmp) {
entry:
  %t1 = zext i16 %tmp2 to i32
  %t2 = shl nuw i32 %t1, 16
  %t3 = zext i16 %tmp1 to i32
  %t4 = or i32 %t2, %t3
  store i32 %t4, i32* %ref.tmp, align 4
  ret void
}

; CHECK-LABEL: int8_int8_pair
; CHECK: movb	%dil, (%rdx)
; CHECK: movb	%sil, 1(%rdx)
define void @int8_int8_pair(i8 signext %tmp1, i8 signext %tmp2, i16* %ref.tmp) {
entry:
  %t1 = zext i8 %tmp2 to i16
  %t2 = shl nuw i16 %t1, 8
  %t3 = zext i8 %tmp1 to i16
  %t4 = or i16 %t2, %t3
  store i16 %t4, i16* %ref.tmp, align 2
  ret void
}

; CHECK-LABEL: int31_int31_pair
; CHECK: andl $2147483647, %edi
; CHECK: movl %edi, (%rdx)
; CHECK: andl $2147483647, %esi
; CHECK: movl %esi, 4(%rdx)
define void @int31_int31_pair(i31 %tmp1, i31 %tmp2, i64* %ref.tmp) {
entry:
  %t1 = zext i31 %tmp2 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i31 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int31_int17_pair
; CHECK: andl $2147483647, %edi
; CHECK: movl %edi, (%rdx)
; CHECK: andl $131071, %esi
; CHECK: movl %esi, 4(%rdx)
define void @int31_int17_pair(i31 %tmp1, i17 %tmp2, i64* %ref.tmp) {
entry:
  %t1 = zext i17 %tmp2 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i31 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: int7_int3_pair
; CHECK: andb $127, %dil
; CHECK: movb %dil, (%rdx)
; CHECK: andb $7, %sil
; CHECK: movb %sil, 1(%rdx)
define void @int7_int3_pair(i7 signext %tmp1, i3 signext %tmp2, i16* %ref.tmp) {
entry:
  %t1 = zext i3 %tmp2 to i16
  %t2 = shl nuw i16 %t1, 8
  %t3 = zext i7 %tmp1 to i16
  %t4 = or i16 %t2, %t3
  store i16 %t4, i16* %ref.tmp, align 2
  ret void
}

; CHECK-LABEL: int24_int24_pair
; CHECK: movw	%di, (%rdx)
; CHECK: shrl	$16, %edi
; CHECK: movb	%dil, 2(%rdx)
; CHECK: movw    %si, 4(%rdx)
; CHECK: shrl    $16, %esi
; CHECK: movb    %sil, 6(%rdx)
define void @int24_int24_pair(i24 signext %tmp1, i24 signext %tmp2, i48* %ref.tmp) {
entry:
  %t1 = zext i24 %tmp2 to i48
  %t2 = shl nuw i48 %t1, 24
  %t3 = zext i24 %tmp1 to i48
  %t4 = or i48 %t2, %t3
  store i48 %t4, i48* %ref.tmp, align 2
  ret void
}

; getTypeSizeInBits(i12) != getTypeStoreSizeInBits(i12), so store split doesn't kick in.
; CHECK-LABEL: int12_int12_pair
; CHECK: movl	%esi, %eax
; CHECK: shll	$12, %eax
; CHECK: andl	$4095, %edi
; CHECK: orl	%eax, %edi
; CHECK: shrl	$4, %esi
; CHECK: movb	%sil, 2(%rdx)
; CHECK: movw	%di, (%rdx)
define void @int12_int12_pair(i12 signext %tmp1, i12 signext %tmp2, i24* %ref.tmp) {
entry:
  %t1 = zext i12 %tmp2 to i24
  %t2 = shl nuw i24 %t1, 12
  %t3 = zext i12 %tmp1 to i24
  %t4 = or i24 %t2, %t3
  store i24 %t4, i24* %ref.tmp, align 2
  ret void
}

; getTypeSizeInBits(i14) != getTypeStoreSizeInBits(i14), so store split doesn't kick in.
; CHECK-LABEL: int7_int7_pair
; CHECK: movzbl	%sil, %eax
; CHECK: shll	$7, %eax
; CHECK: andb	$127, %dil
; CHECK: movzbl	%dil, %ecx
; CHECK: orl	%eax, %ecx
; CHECK: andl	$16383, %ecx
; CHECK: movw	%cx, (%rdx)
define void @int7_int7_pair(i7 signext %tmp1, i7 signext %tmp2, i14* %ref.tmp) {
entry:
  %t1 = zext i7 %tmp2 to i14
  %t2 = shl nuw i14 %t1, 7
  %t3 = zext i7 %tmp1 to i14
  %t4 = or i14 %t2, %t3
  store i14 %t4, i14* %ref.tmp, align 2
  ret void
}

; getTypeSizeInBits(i2) != getTypeStoreSizeInBits(i2), so store split doesn't kick in.
; CHECK-LABEL: int1_int1_pair
; CHECK: addb %sil, %sil
; CHECK: andb $1, %dil
; CHECK: orb %sil, %dil
; CHECK: andb $3, %dil
; CHECK: movb %dil, (%rdx)
define void @int1_int1_pair(i1 signext %tmp1, i1 signext %tmp2, i2* %ref.tmp) {
entry:
  %t1 = zext i1 %tmp2 to i2
  %t2 = shl nuw i2 %t1, 1
  %t3 = zext i1 %tmp1 to i2
  %t4 = or i2 %t2, %t3
  store i2 %t4, i2* %ref.tmp, align 1
  ret void
}

; CHECK-LABEL: mbb_int32_float_pair
; CHECK: movl %edi, (%rsi)
; CHECK: movss %xmm0, 4(%rsi)
define void @mbb_int32_float_pair(i32 %tmp1, float %tmp2, i64* %ref.tmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  br label %next
next:
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i32 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  ret void
}

; CHECK-LABEL: mbb_int32_float_multi_stores
; CHECK: movl %edi, (%rsi)
; CHECK: movss %xmm0, 4(%rsi)
; CHECK: # %bb2
; CHECK: movl %edi, (%rdx)
; CHECK: movss %xmm0, 4(%rdx)
define void @mbb_int32_float_multi_stores(i32 %tmp1, float %tmp2, i64* %ref.tmp, i64* %ref.tmp1, i1 %cmp) {
entry:
  %t0 = bitcast float %tmp2 to i32
  br label %bb1
bb1:
  %t1 = zext i32 %t0 to i64
  %t2 = shl nuw i64 %t1, 32
  %t3 = zext i32 %tmp1 to i64
  %t4 = or i64 %t2, %t3
  store i64 %t4, i64* %ref.tmp, align 8
  br i1 %cmp, label %bb2, label %exitbb
bb2:
  store i64 %t4, i64* %ref.tmp1, align 8
  br label %exitbb
exitbb:
  ret void
}
