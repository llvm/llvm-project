; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;  void multiple_types(i8 *A) {
;    for (long i = 0; i < 100; i++) {
;      A[i] = *(i1 *)&A[1 * i] +
;             *(i16 *)&A[2 * i] +
;             *(i24 *)&A[4 * i] +
;             *(i32 *)&A[4 * i] +
;             *(i40 *)&A[8 * i] +
;             *(i48 *)&A[8 * i] +
;             *(i56 *)&A[8 * i] +
;             *(i64 *)&A[8 * i] +
;             *(i120 *)&A[16 * i] +
;             *(i192 *)&A[24 * i] +
;             *(i248 *)&A[32 * i];
;    }
;  }
;
; Verify that different data type sizes are correctly modeled. Specifically,
; we want to verify that type i1 is modeled with allocation size i8,
; type i24 is modeled with allocation size i32 and that i40, i48 and i56 are
; modeled with allocation size i64. Larger types, e.g., i120, i192 and i248 are
; not rounded up to the next power-of-two allocation size, but rather to the
; next multiple of 64.

; The allocation size discussed above defines the number of canonical array
; elements accessed. For example, even though i24 only consists of 3 bytes,
; its allocation size is 4 bytes. Consequently, we model the access to an
; i24 element as an access to four canonical elements resulting in access
; relation constraints '4i0 <= o0 <= 3 + 4i0' instead of '3i0 <= o0 <= 2 + 3i0'.

; CHECK: Statements {
; CHECK:   Stmt_bb2
; CHECK:         Domain :=
; CHECK:             { Stmt_bb2[i0] : 0 <= i0 <= 99 };
; CHECK:         Schedule :=
; CHECK:             { Stmt_bb2[i0] -> [i0] };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 2i0 <= o0 <= 1 + 2i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 4i0 <= o0 <= 3 + 4i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 4i0 <= o0 <= 3 + 4i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 8i0 <= o0 <= 7 + 8i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 8i0 <= o0 <= 7 + 8i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 8i0 <= o0 <= 7 + 8i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 8i0 <= o0 <= 7 + 8i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 16i0 <= o0 <= 15 + 16i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 24i0 <= o0 <= 23 + 24i0 };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 32i0 <= o0 <= 31 + 32i0 };
; CHECK:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[i0] };
; CHECK: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multiple_types(i8* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb20, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp21, %bb20 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb22

bb2:                                              ; preds = %bb1
  %load.i1.offset = mul i64 %i.0, 1
  %load.i1.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i1.offset
  %load.i1.ptrcast = bitcast i8* %load.i1.ptr to i1*
  %load.i1.val = load i1, i1* %load.i1.ptrcast
  %load.i1.val.trunc = zext i1 %load.i1.val to i8

  %load.i16.offset = mul i64 %i.0, 2
  %load.i16.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i16.offset
  %load.i16.ptrcast = bitcast i8* %load.i16.ptr to i16*
  %load.i16.val = load i16, i16* %load.i16.ptrcast
  %load.i16.val.trunc = trunc i16 %load.i16.val to i8

  %load.i24.offset = mul i64 %i.0, 4
  %load.i24.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i24.offset
  %load.i24.ptrcast = bitcast i8* %load.i24.ptr to i24*
  %load.i24.val = load i24, i24* %load.i24.ptrcast
  %load.i24.val.trunc = trunc i24 %load.i24.val to i8

  %load.i32.offset = mul i64 %i.0, 4
  %load.i32.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i32.offset
  %load.i32.ptrcast = bitcast i8* %load.i32.ptr to i32*
  %load.i32.val = load i32, i32* %load.i32.ptrcast
  %load.i32.val.trunc = trunc i32 %load.i32.val to i8

  %load.i40.offset = mul i64 %i.0, 8
  %load.i40.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i40.offset
  %load.i40.ptrcast = bitcast i8* %load.i40.ptr to i40*
  %load.i40.val = load i40, i40* %load.i40.ptrcast
  %load.i40.val.trunc = trunc i40 %load.i40.val to i8

  %load.i48.offset = mul i64 %i.0, 8
  %load.i48.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i48.offset
  %load.i48.ptrcast = bitcast i8* %load.i48.ptr to i48*
  %load.i48.val = load i48, i48* %load.i48.ptrcast
  %load.i48.val.trunc = trunc i48 %load.i48.val to i8

  %load.i56.offset = mul i64 %i.0, 8
  %load.i56.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i56.offset
  %load.i56.ptrcast = bitcast i8* %load.i56.ptr to i56*
  %load.i56.val = load i56, i56* %load.i56.ptrcast
  %load.i56.val.trunc = trunc i56 %load.i56.val to i8

  %load.i64.offset = mul i64 %i.0, 8
  %load.i64.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i64.offset
  %load.i64.ptrcast = bitcast i8* %load.i64.ptr to i64*
  %load.i64.val = load i64, i64* %load.i64.ptrcast
  %load.i64.val.trunc = trunc i64 %load.i64.val to i8

  %load.i120.offset = mul i64 %i.0, 16
  %load.i120.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i120.offset
  %load.i120.ptrcast = bitcast i8* %load.i120.ptr to i120*
  %load.i120.val = load i120, i120* %load.i120.ptrcast
  %load.i120.val.trunc = trunc i120 %load.i120.val to i8

  %load.i192.offset = mul i64 %i.0, 24
  %load.i192.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i192.offset
  %load.i192.ptrcast = bitcast i8* %load.i192.ptr to i192*
  %load.i192.val = load i192, i192* %load.i192.ptrcast
  %load.i192.val.trunc = trunc i192 %load.i192.val to i8

  %load.i248.offset = mul i64 %i.0, 32
  %load.i248.ptr = getelementptr inbounds i8, i8* %A, i64 %load.i248.offset
  %load.i248.ptrcast = bitcast i8* %load.i248.ptr to i248*
  %load.i248.val = load i248, i248* %load.i248.ptrcast
  %load.i248.val.trunc = trunc i248 %load.i248.val to i8

  %sum = add i8 %load.i1.val.trunc, %load.i16.val.trunc
  %sum0 = add i8 %sum, %load.i24.val.trunc
  %sum1 = add i8 %sum0, %load.i32.val.trunc
  %sum2 = add i8 %sum1, %load.i40.val.trunc
  %sum3 = add i8 %sum2, %load.i48.val.trunc
  %sum4 = add i8 %sum3, %load.i56.val.trunc
  %sum5 = add i8 %sum4, %load.i64.val.trunc
  %sum6 = add i8 %sum5, %load.i120.val.trunc
  %sum7 = add i8 %sum6, %load.i192.val.trunc
  %sum8 = add i8 %sum7, %load.i248.val.trunc
  %tmp7 = getelementptr inbounds i8, i8* %A, i64 %i.0
  store i8 %sum8, i8* %tmp7
  br label %bb20

bb20:                                             ; preds = %bb2
  %tmp21 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb22:                                             ; preds = %bb1
  ret void
}
