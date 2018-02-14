; RUN: opt %loadPolly -polly-scops -analyze -polly-allow-modref-calls \
; RUN: -polly-invariant-load-hoisting=true \
; RUN: < %s | FileCheck %s
; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine -analyze \
; RUN: -polly-invariant-load-hoisting=true \
; RUN: -polly-allow-modref-calls < %s | FileCheck %s --check-prefix=NONAFFINE

;  TODO: We should delinearize the accesses despite the use in a call to a
;        readonly function. For now we verify we do not delinearize them though.

; CHECK:         Function: ham
; CHECK-NEXT:    Region: %bb12---%bb28
; CHECK-NEXT:    Max Loop Depth:  1
; CHECK-NEXT:    Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb12[] -> MemRef_arg1[0] };
; CHECK-NEXT:            Execution Context: [tmp14, p_1] -> {  :  }
; CHECK-NEXT:    }
; CHECK-NEXT:    Context:
; CHECK-NEXT:    [tmp14, p_1] -> {  : -9223372036854775808 <= tmp14 <= 9223372036854775807 and -9223372036854775808 <= p_1 <= 9223372036854775807 }
; CHECK-NEXT:    Assumed Context:
; CHECK-NEXT:    [tmp14, p_1] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [tmp14, p_1] -> { : tmp14 > 0 and (p_1 <= -1152921504606846977 or tmp14 >= 1152921504606846977 or p_1 >= 1152921504606846977 - tmp14) }
; CHECK-NEXT:    p0: %tmp14
; CHECK-NEXT:    p1: {0,+,(0 smax %tmp)}<%bb12>
; CHECK-NEXT:    Arrays {
; CHECK-NEXT:        i64 MemRef_arg1[*]; // Element size 8
; CHECK-NEXT:        i64 MemRef_tmp13; // Element size 8
; CHECK-NEXT:        double MemRef_arg4[*]; // Element size 8
; CHECK-NEXT:    }
; CHECK-NEXT:    Arrays (Bounds as pw_affs) {
; CHECK-NEXT:        i64 MemRef_arg1[*]; // Element size 8
; CHECK-NEXT:        i64 MemRef_tmp13; // Element size 8
; CHECK-NEXT:        double MemRef_arg4[*]; // Element size 8
; CHECK-NEXT:    }
; CHECK-NEXT:    Alias Groups (0):
; CHECK-NEXT:        n/a
; CHECK-NEXT:    Statements {
; CHECK-NEXT:    	Stmt_bb12
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb12[] };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb12[] -> [0, 0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb12[] -> MemRef_tmp13[] };
; CHECK-NEXT:    	Stmt_bb17
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb17[i0] : 0 <= i0 < tmp14 };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb17[i0] -> [1, i0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb17[i0] -> MemRef_arg4[p_1 + i0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb17[i0] -> MemRef_arg1[o0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [tmp14, p_1] -> { Stmt_bb17[i0] -> MemRef_arg4[o0] };
; CHECK-NEXT:    }


; NONAFFINE:         Function: ham
; NONAFFINE-NEXT:    Region: %bb5---%bb32
; NONAFFINE-NEXT:    Max Loop Depth:  2
; NONAFFINE-NEXT:    Invariant Accesses: {
; NONAFFINE-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb5[] -> MemRef_arg[0] };
; NONAFFINE-NEXT:            Execution Context: [tmp9, tmp14] -> {  :  }
; NONAFFINE-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb12[i0] -> MemRef_arg1[0] };
; NONAFFINE-NEXT:            Execution Context: [tmp9, tmp14] -> {  :  }
; NONAFFINE-NEXT:    }
; NONAFFINE-NEXT:    Context:
; NONAFFINE-NEXT:    [tmp9, tmp14] -> {  : -9223372036854775808 <= tmp9 <= 9223372036854775807 and -9223372036854775808 <= tmp14 <= 9223372036854775807 }
; NONAFFINE-NEXT:    Assumed Context:
; NONAFFINE-NEXT:    [tmp9, tmp14] -> {  :  }
; NONAFFINE-NEXT:    Invalid Context:
; NONAFFINE-NEXT:    [tmp9, tmp14] -> {  : 1 = 0 }
; NONAFFINE-NEXT:    p0: %tmp9
; NONAFFINE-NEXT:    p1: %tmp14
; NONAFFINE-NEXT:    Arrays {
; NONAFFINE-NEXT:        i64 MemRef_arg[*]; // Element size 8
; NONAFFINE-NEXT:        i64 MemRef_arg1[*]; // Element size 8
; NONAFFINE-NEXT:        i64 MemRef_tmp7; // Element size 8
; NONAFFINE-NEXT:        i64 MemRef_tmp8; // Element size 8
; NONAFFINE-NEXT:        double MemRef_arg4[*]; // Element size 8
; NONAFFINE-NEXT:    }
; NONAFFINE-NEXT:    Arrays (Bounds as pw_affs) {
; NONAFFINE-NEXT:        i64 MemRef_arg[*]; // Element size 8
; NONAFFINE-NEXT:        i64 MemRef_arg1[*]; // Element size 8
; NONAFFINE-NEXT:        i64 MemRef_tmp7; // Element size 8
; NONAFFINE-NEXT:        i64 MemRef_tmp8; // Element size 8
; NONAFFINE-NEXT:        double MemRef_arg4[*]; // Element size 8
; NONAFFINE-NEXT:    }
; NONAFFINE-NEXT:    Alias Groups (0):
; NONAFFINE-NEXT:        n/a
; NONAFFINE-NEXT:    Statements {
; NONAFFINE-NEXT:    	Stmt_bb5
; NONAFFINE-NEXT:            Domain :=
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb5[] };
; NONAFFINE-NEXT:            Schedule :=
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb5[] -> [0, 0, 0] };
; NONAFFINE-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb5[] -> MemRef_tmp7[] };
; NONAFFINE-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb5[] -> MemRef_tmp8[] };
; NONAFFINE-NEXT:    	Stmt_bb17
; NONAFFINE-NEXT:            Domain :=
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] : 0 <= i0 < tmp9 and 0 <= i1 < tmp14 };
; NONAFFINE-NEXT:            Schedule :=
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> [1, i0, i1] };
; NONAFFINE-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> MemRef_tmp7[] };
; NONAFFINE-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> MemRef_tmp8[] };
; NONAFFINE-NEXT:            MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> MemRef_arg4[o0] };
; NONAFFINE-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> MemRef_arg[o0] };
; NONAFFINE-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> MemRef_arg1[o0] };
; NONAFFINE-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:                [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> MemRef_arg4[o0] };

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @ham(i64* noalias %arg, i64* noalias %arg1, i64* noalias %arg2, i64* noalias %arg3, [1000 x double]* noalias %arg4) unnamed_addr {
bb:
  br label %bb5

bb5:                                              ; preds = %bb
  %tmp = load i64, i64* %arg1, align 8
  %tmp6 = icmp slt i64 %tmp, 0
  %tmp7 = select i1 %tmp6, i64 0, i64 %tmp
  %tmp8 = xor i64 %tmp7, -1
  %tmp9 = load i64, i64* %arg, align 8
  %tmp10 = icmp sgt i64 %tmp9, 0
  br i1 %tmp10, label %bb11, label %bb32

bb11:                                             ; preds = %bb5
  br label %bb12

bb12:                                             ; preds = %bb28, %bb11
  %tmp13 = phi i64 [ %tmp30, %bb28 ], [ 1, %bb11 ]
  %tmp14 = load i64, i64* %arg1, align 8
  %tmp15 = icmp sgt i64 %tmp14, 0
  br i1 %tmp15, label %bb16, label %bb28

bb16:                                             ; preds = %bb12
  br label %bb17

bb17:                                             ; preds = %bb17, %bb16
  %tmp18 = phi i64 [ %tmp26, %bb17 ], [ 1, %bb16 ]
  %tmp19 = mul i64 %tmp13, %tmp7
  %tmp20 = add i64 %tmp19, %tmp8
  %tmp21 = add i64 %tmp20, %tmp18
  %tmp22 = add i64 %tmp18, %tmp13
  %tmp23 = sitofp i64 %tmp22 to double
  %tmp24 = getelementptr [1000 x double], [1000 x double]* %arg4, i64 0, i64 %tmp21
  %call = call double @func(double* %tmp24) #2
  %sum = fadd double %call, %tmp23
  store double %sum, double* %tmp24, align 8
  %tmp25 = icmp eq i64 %tmp18, %tmp14
  %tmp26 = add i64 %tmp18, 1
  br i1 %tmp25, label %bb27, label %bb17

bb27:                                             ; preds = %bb17
  br label %bb28

bb28:                                             ; preds = %bb27, %bb12
  %tmp29 = icmp eq i64 %tmp13, %tmp9
  %tmp30 = add i64 %tmp13, 1
  br i1 %tmp29, label %bb31, label %bb12

bb31:                                             ; preds = %bb28
  br label %bb32

bb32:                                             ; preds = %bb31, %bb5
  ret void
}

declare double @func(double*) #1

attributes #1 = { nounwind readonly }
