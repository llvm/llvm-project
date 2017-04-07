; RUN: llc < %s -mtriple=arm-eabi -mcpu=krait | FileCheck %s

define void @func1(i16* %a, i16* %b, i16* %c) {
entry:
; The test case trying to vectorize the pseudo code below.
; a[i] = b[i] + c[i];
; b[i] = a[i] * c[i];
; a[i] = b[i] + a[i] * c[i];
;
; Checking that vector load a[i] for "a[i] = b[i] + a[i] * c[i]" is
; scheduled before the first vector store to "a[i] = b[i] + c[i]".
; Checking that there is no vector load a[i] scheduled between the vector
; stores to a[i], otherwise the load of a[i] will be polluted by the first
; vector store to a[i].
;
; This test case check that the chain information is updated during
; lowerMUL for the new created Load SDNode.

; CHECK: vldr {{.*}} [r0, #16]
; CHECK: vstr {{.*}} [r0, #16]
; CHECK-NOT: vldr {{.*}} [r0, #16]
; CHECK: vstr {{.*}} [r0, #16]

  %scevgep0 = getelementptr i16, i16* %a, i32 8
  %vector_ptr0 = bitcast i16* %scevgep0 to <4 x i16>*
  %vec0 = load <4 x i16>, <4 x i16>* %vector_ptr0, align 8
  %scevgep1 = getelementptr i16, i16* %b, i32 8
  %vector_ptr1 = bitcast i16* %scevgep1 to <4 x i16>*
  %vec1 = load <4 x i16>, <4 x i16>* %vector_ptr1, align 8
  %0 = zext <4 x i16> %vec1 to <4 x i32>
  %scevgep2 = getelementptr i16, i16* %c, i32 8
  %vector_ptr2 = bitcast i16* %scevgep2 to <4 x i16>*
  %vec2 = load <4 x i16>, <4 x i16>* %vector_ptr2, align 8
  %1 = sext <4 x i16> %vec2 to <4 x i32>
  %vec3 = add <4 x i32> %1, %0
  %2 = trunc <4 x i32> %vec3 to <4 x i16>
  %scevgep3 = getelementptr i16, i16* %a, i32 8
  %vector_ptr3 = bitcast i16* %scevgep3 to <4 x i16>*
  store <4 x i16> %2, <4 x i16>* %vector_ptr3, align 8
  %vector_ptr4 = bitcast i16* %scevgep2 to <4 x i16>*
  %vec4 = load <4 x i16>, <4 x i16>* %vector_ptr4, align 8
  %3 = sext <4 x i16> %vec4 to <4 x i32>
  %vec5 = mul <4 x i32> %3, %vec3
  %4 = trunc <4 x i32> %vec5 to <4 x i16>
  %vector_ptr5 = bitcast i16* %scevgep1 to <4 x i16>*
  store <4 x i16> %4, <4 x i16>* %vector_ptr5, align 8
  %5 = sext <4 x i16> %vec0 to <4 x i32>
  %vector_ptr6 = bitcast i16* %scevgep2 to <4 x i16>*
  %vec6 = load <4 x i16>, <4 x i16>* %vector_ptr6, align 8
  %6 = sext <4 x i16> %vec6 to <4 x i32>
  %vec7 = mul <4 x i32> %6, %5
  %vec8 = add <4 x i32> %vec7, %vec5
  %7 = trunc <4 x i32> %vec8 to <4 x i16>
  %vector_ptr7 = bitcast i16* %scevgep3 to <4 x i16>*
  store <4 x i16> %7, <4 x i16>* %vector_ptr7, align 8
  ret void
}

define void @func2(i16* %a, i16* %b, i16* %c) {
entry:
; The test case trying to vectorize the pseudo code below.
; a[i] = b[i] + c[i];
; b[i] = a[i] * c[i];
; a[i] = b[i] + a[i] * c[i] + a[i];
;
; Checking that vector load a[i] for "a[i] = b[i] + a[i] * c[i] + a[i]"
; is scheduled before the first vector store to "a[i] = b[i] + c[i]".
; Checking that there is no vector load a[i] scheduled between the first
; vector store to a[i] and the vector add of a[i], otherwise the load of
; a[i] will be polluted by the first vector store to a[i].
;
; This test case check that both the chain and value of the new created
; Load SDNode are updated during lowerMUL.

; CHECK: vldr {{.*}} [r0, #16]
; CHECK: vstr {{.*}} [r0, #16]
; CHECK-NOT: vldr {{.*}} [r0, #16]
; CHECK: vaddw.s16
; CHECK: vstr {{.*}} [r0, #16]

  %scevgep0 = getelementptr i16, i16* %a, i32 8
  %vector_ptr0 = bitcast i16* %scevgep0 to <4 x i16>*
  %vec0 = load <4 x i16>, <4 x i16>* %vector_ptr0, align 8
  %scevgep1 = getelementptr i16, i16* %b, i32 8
  %vector_ptr1 = bitcast i16* %scevgep1 to <4 x i16>*
  %vec1 = load <4 x i16>, <4 x i16>* %vector_ptr1, align 8
  %0 = zext <4 x i16> %vec1 to <4 x i32>
  %scevgep2 = getelementptr i16, i16* %c, i32 8
  %vector_ptr2 = bitcast i16* %scevgep2 to <4 x i16>*
  %vec2 = load <4 x i16>, <4 x i16>* %vector_ptr2, align 8
  %1 = sext <4 x i16> %vec2 to <4 x i32>
  %vec3 = add <4 x i32> %1, %0
  %2 = trunc <4 x i32> %vec3 to <4 x i16>
  %scevgep3 = getelementptr i16, i16* %a, i32 8
  %vector_ptr3 = bitcast i16* %scevgep3 to <4 x i16>*
  store <4 x i16> %2, <4 x i16>* %vector_ptr3, align 8
  %vector_ptr4 = bitcast i16* %scevgep2 to <4 x i16>*
  %vec4 = load <4 x i16>, <4 x i16>* %vector_ptr4, align 8
  %3 = sext <4 x i16> %vec4 to <4 x i32>
  %vec5 = mul <4 x i32> %3, %vec3
  %4 = trunc <4 x i32> %vec5 to <4 x i16>
  %vector_ptr5 = bitcast i16* %scevgep1 to <4 x i16>*
  store <4 x i16> %4, <4 x i16>* %vector_ptr5, align 8
  %5 = sext <4 x i16> %vec0 to <4 x i32>
  %vector_ptr6 = bitcast i16* %scevgep2 to <4 x i16>*
  %vec6 = load <4 x i16>, <4 x i16>* %vector_ptr6, align 8
  %6 = sext <4 x i16> %vec6 to <4 x i32>
  %vec7 = mul <4 x i32> %6, %5
  %vec8 = add <4 x i32> %vec7, %vec5
  %vec9 = add <4 x i32> %vec8, %5
  %7 = trunc <4 x i32> %vec9 to <4 x i16>
  %vector_ptr7 = bitcast i16* %scevgep3 to <4 x i16>*
  store <4 x i16> %7, <4 x i16>* %vector_ptr7, align 8
  ret void
}
