; RUN: split-file %s %t
; RUN: opt -S -passes=vector-combine -verify-each -mtriple=x86_64-unknown-linux-gnu < %t/load.ll  | FileCheck %s --check-prefix=LOAD
; RUN: opt -S -passes=vector-combine -verify-each -mtriple=x86_64-unknown-linux-gnu < %t/store.ll | FileCheck %s --check-prefix=STORE
; RUN: opt -S -passes=vector-combine -verify-each -mtriple=x86_64-unknown-linux-gnu < %t/packed-stride.ll | FileCheck %s --check-prefix=PACKED
; RUN: opt -S -passes=vector-combine -verify-each -mtriple=x86_64-unknown-linux-gnu < %t/p32-index.ll | FileCheck %s --check-prefix=P32
; RUN: opt -S -passes=vector-combine -verify-each -mtriple=x86_64-unknown-linux-gnu < %t/p32-load-index.ll | FileCheck %s --check-prefix=P32LOAD

;--- load.ll
target datalayout = "e-p:64:64:64:8"

; The lane range fits in the vector index, but its maximum byte offset does not
; fit in the signed GEP index type. The transform must reject the candidate
; without leaving a freeze behind or asserting on the pending freeze state.
define i16 @load_extract_unrepresentable_offset(ptr %p, i8 %idx) {
; LOAD-LABEL: @load_extract_unrepresentable_offset(
; LOAD-NEXT:    [[BOUNDED:%.*]] = and i8 [[IDX:%.*]], 127
; LOAD-NEXT:    [[V:%.*]] = load <128 x i16>, ptr [[P:%.*]], align 2
; LOAD-NEXT:    [[X:%.*]] = extractelement <128 x i16> [[V]], i8 [[BOUNDED]]
; LOAD-NEXT:    ret i16 [[X]]
;
  %bounded = and i8 %idx, 127
  %v = load <128 x i16>, ptr %p, align 2
  %x = extractelement <128 x i16> %v, i8 %bounded
  ret i16 %x
}

;--- store.ll
target datalayout = "e-p:64:64:64:8"

define void @insert_store_unrepresentable_offset(ptr %p, i16 %x, i8 %idx) {
; STORE-LABEL: @insert_store_unrepresentable_offset(
; STORE-NEXT:    [[BOUNDED:%.*]] = and i8 [[IDX:%.*]], 127
; STORE-NEXT:    [[V:%.*]] = load <128 x i16>, ptr [[P:%.*]], align 2
; STORE-NEXT:    [[V1:%.*]] = insertelement <128 x i16> [[V]], i16 [[X:%.*]], i8 [[BOUNDED]]
; STORE-NEXT:    store <128 x i16> [[V1]], ptr [[P]], align 2
; STORE-NEXT:    ret void
;
  %bounded = and i8 %idx, 127
  %v = load <128 x i16>, ptr %p, align 2
  %v1 = insertelement <128 x i16> %v, i16 %x, i8 %bounded
  store <128 x i16> %v1, ptr %p, align 2
  ret void
}

;--- packed-stride.ll
target datalayout = "e-p:64:64:64:8-i24:32:32"

; Vector elements are tightly packed. For i24, the 3-byte element store size
; keeps the maximum offset (42 * 3) representable by the signed i8 GEP index;
; the 4-byte ABI allocation size does not.
define void @insert_store_packed_i24_stride(ptr %p, i24 %x, i6 %idx) {
; PACKED-LABEL: define void @insert_store_packed_i24_stride(
; PACKED-SAME: ptr [[P:%.*]], i24 [[X:%.*]], i6 [[IDX:%.*]]) {
; PACKED-NEXT:    [[IDX_FROZEN:%.*]] = freeze i6 [[IDX]]
; PACKED-NEXT:    [[BOUNDED:%.*]] = urem i6 [[IDX_FROZEN]], -21
; PACKED-NEXT:    [[BOUNDED_GEPIDX:%.*]] = zext i6 [[BOUNDED]] to i8
; PACKED-NEXT:    [[TMP1:%.*]] = getelementptr inbounds <43 x i24>, ptr [[P]], i8 0, i8 [[BOUNDED_GEPIDX]]
; PACKED-NEXT:    store i24 [[X]], ptr [[TMP1]], align 1
; PACKED-NEXT:    ret void
;
  %bounded = urem i6 %idx, 43
  %v = load <43 x i24>, ptr %p, align 1
  %v1 = insertelement <43 x i24> %v, i24 %x, i6 %bounded
  store <43 x i24> %v1, ptr %p, align 1
  ret void
}

;--- p32-index.ll
target datalayout = "e-p:64:64:64:32"

; A pointer's index type, rather than its pointer width, determines the
; zero-extension destination type.
define void @insert_store_dynamic_p32_index(ptr %p, i8 %x, i4 %idx) {
; P32-LABEL: define void @insert_store_dynamic_p32_index(
; P32-SAME: ptr [[P:%.*]], i8 [[X:%.*]], i4 [[IDX:%.*]]) {
; P32-NEXT:    [[IDX_FROZEN:%.*]] = freeze i4 [[IDX]]
; P32-NEXT:    [[BOUNDED:%.*]] = urem i4 [[IDX_FROZEN]], -1
; P32-NEXT:    [[BOUNDED_GEPIDX:%.*]] = zext i4 [[BOUNDED]] to i32
; P32-NEXT:    [[TMP1:%.*]] = getelementptr inbounds <15 x i8>, ptr [[P]], i32 0, i32 [[BOUNDED_GEPIDX]]
; P32-NEXT:    store i8 [[X]], ptr [[TMP1]], align 1
; P32-NEXT:    ret void
;
  %bounded = urem i4 %idx, 15
  %v = load <15 x i8>, ptr %p, align 1
  %v1 = insertelement <15 x i8> %v, i8 %x, i4 %bounded
  store <15 x i8> %v1, ptr %p, align 1
  ret void
}

; A vector of pointers reaches getScalarizedGEPIndexInfo as VecTy through the
; insert/store path. PtrTy remains scalar, as required by load and store.
define void @insert_store_pointer_vector_dynamic_p32_index(ptr %p, ptr %x, i4 %idx) {
; P32-LABEL: define void @insert_store_pointer_vector_dynamic_p32_index(
; P32-SAME: ptr [[P:%.*]], ptr [[X:%.*]], i4 [[IDX:%.*]]) {
; P32-NEXT:    [[IDX_FROZEN:%.*]] = freeze i4 [[IDX]]
; P32-NEXT:    [[BOUNDED:%.*]] = urem i4 [[IDX_FROZEN]], -1
; P32-NEXT:    [[BOUNDED_GEPIDX:%.*]] = zext i4 [[BOUNDED]] to i32
; P32-NEXT:    [[TMP1:%.*]] = getelementptr inbounds <15 x ptr>, ptr [[P]], i32 0, i32 [[BOUNDED_GEPIDX]]
; P32-NEXT:    store ptr [[X]], ptr [[TMP1]], align 8
; P32-NEXT:    ret void
;
  %bounded = urem i4 %idx, 15
  %v = load <15 x ptr>, ptr %p, align 8
  %v1 = insertelement <15 x ptr> %v, ptr %x, i4 %bounded
  store <15 x ptr> %v1, ptr %p, align 8
  ret void
}

; The i4 bit pattern -2 denotes unsigned lane 14. Materializing it for an i32
; GEP index must not create an instruction or reinterpret it as a negative
; signed index.
define void @insert_store_constant_high_bit_p32_index(ptr %p, i8 %x) {
; P32-LABEL: define void @insert_store_constant_high_bit_p32_index(
; P32-SAME: ptr [[P:%.*]], i8 [[X:%.*]]) {
; P32-NEXT:    [[GEP:%.*]] = getelementptr inbounds <15 x i8>, ptr [[P]], i32 0, i32 14
; P32-NEXT:    store i8 [[X]], ptr [[GEP]], align 1
; P32-NEXT:    ret void
;
  %v = load <15 x i8>, ptr %p, align 1
  %v1 = insertelement <15 x i8> %v, i8 %x, i4 -2
  store <15 x i8> %v1, ptr %p, align 1
  ret void
}

;--- p32-load-index.ll
target datalayout = "e-p:64:64:64:32"

; scalarizeLoadExtract records index information before it replaces each
; extract. The lookup must recover the 32-bit GEP index type for this extract.
define i8 @load_extract_dynamic_p32_index(ptr %p, i4 %idx) {
; P32LOAD-LABEL: define i8 @load_extract_dynamic_p32_index(
; P32LOAD-SAME: ptr [[P:%.*]], i4 [[IDX:%.*]]) {
; P32LOAD-NEXT:    [[IDX_FROZEN:%.*]] = freeze i4 [[IDX]]
; P32LOAD-NEXT:    [[BOUNDED:%.*]] = urem i4 [[IDX_FROZEN]], -1
; P32LOAD-NEXT:    [[BOUNDED_GEPIDX:%.*]] = zext i4 [[BOUNDED]] to i32
; P32LOAD-NEXT:    [[GEP:%.*]] = getelementptr inbounds <15 x i8>, ptr [[P]], i32 0, i32 [[BOUNDED_GEPIDX]]
; P32LOAD-NEXT:    [[X:%.*]] = load i8, ptr [[GEP]], align 1
; P32LOAD-NEXT:    ret i8 [[X]]
;
  %bounded = urem i4 %idx, 15
  %v = load <15 x i8>, ptr %p, align 1
  %x = extractelement <15 x i8> %v, i4 %bounded
  ret i8 %x
}

; A vector of pointers reaches getScalarizedGEPIndexInfo as VecTy through the
; load/extract path. PtrTy remains scalar, as required by load instructions.
define ptr @load_extract_pointer_vector_dynamic_p32_index(ptr %p, i4 %idx) {
; P32LOAD-LABEL: define ptr @load_extract_pointer_vector_dynamic_p32_index(
; P32LOAD-SAME: ptr [[P:%.*]], i4 [[IDX:%.*]]) {
; P32LOAD-NEXT:    [[IDX_FROZEN:%.*]] = freeze i4 [[IDX]]
; P32LOAD-NEXT:    [[BOUNDED:%.*]] = urem i4 [[IDX_FROZEN]], -1
; P32LOAD-NEXT:    [[BOUNDED_GEPIDX:%.*]] = zext i4 [[BOUNDED]] to i32
; P32LOAD-NEXT:    [[GEP:%.*]] = getelementptr inbounds <15 x ptr>, ptr [[P]], i32 0, i32 [[BOUNDED_GEPIDX]]
; P32LOAD-NEXT:    [[X:%.*]] = load ptr, ptr [[GEP]], align 8
; P32LOAD-NEXT:    ret ptr [[X]]
;
  %bounded = urem i4 %idx, 15
  %v = load <15 x ptr>, ptr %p, align 8
  %x = extractelement <15 x ptr> %v, i4 %bounded
  ret ptr %x
}
