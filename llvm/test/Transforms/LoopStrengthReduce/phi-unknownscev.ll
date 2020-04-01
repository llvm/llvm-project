; RUN: llc -debug-only=loop-reduce < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
 
%_elem_type_of_x = type <{ float }>
%_elem_type_of_y = type <{ float }>
%_elem_type_of_ap = type <{ float }>
%_elem_type_of_aux = type <{ float }>

@zero = internal global float 0.000000e+00, align 4

; Should not elimimate %lsr.iv30, it is not congruent with %lsr.iv99.
; Their start value are not same.
; %scevgep26 = getelementptr [0 x %_elem_type_of_ap], [0 x %_elem_type_of_ap]* %.ap, i64 0, i64 28
; %scevgep2627 = bitcast %_elem_type_of_ap* %scevgep26 to [0 x %_elem_type_of_ap]*
; %lsr.iv30 = phi [0 x %_elem_type_of_ap]* [ %24, %_scf_2_skip_ ], [ %scevgep2627, %_loop_1_do_.lr.ph ]
;
; %scevgep97 = getelementptr [0 x %_elem_type_of_ap], [0 x %_elem_type_of_ap]* %.ap, i64 0, i64 -1
; %scevgep9798 = bitcast %_elem_type_of_ap* %scevgep97 to [0 x %_elem_type_of_ap]*
; %lsr.iv99 = phi [0 x %_elem_type_of_ap]* [ %24, %_scf_2_skip_ ], [ %scevgep9798, %_loop_1_do_.lr.ph ]

; CHECK: INDVARS: Eliminated congruent iv:
; CHECK-NEXT: INDVARS: Original iv:

define void @foo(i64* noalias %.n, float* noalias %.alpha, [0 x %_elem_type_of_x]* noalias %.x, [0 x %_elem_type_of_y]* noalias %.y, [0 x %_elem_type_of_ap]* noalias %.ap, i64* noalias %.lda) {
entry:
  %_val_n_ = load i64, i64* %.n, align 8
  %_mod_tmp = srem i64 %_val_n_, 2
  %_add_tmp = add nsw i64 %_mod_tmp, 16
  %_les_tmp = icmp slt i64 %_val_n_, %_add_tmp
  %min = select i1 %_les_tmp, i64 %_val_n_, i64 %_add_tmp
  %_grt_tmp20 = icmp slt i64 %min, 1
  br i1 %_grt_tmp20, label %_return_bb, label %_loop_1_do_.lr.ph

_loop_1_do_.lr.ph:                                ; preds = %entry
  %_val_lda_ = load i64, i64* %.lda, align 8
  %x_rvo_based_addr_11 = getelementptr inbounds [0 x %_elem_type_of_x], [0 x %_elem_type_of_x]* %.x, i64 0, i64 -1
  %y_rvo_based_addr_13 = getelementptr inbounds [0 x %_elem_type_of_y], [0 x %_elem_type_of_y]* %.y, i64 0, i64 -1
  %ap_rvo_based_addr_15 = getelementptr inbounds [0 x %_elem_type_of_ap], [0 x %_elem_type_of_ap]* %.ap, i64 0, i64 -1
  %_add_tmp55 = sub i64 %_val_lda_, %_val_n_
  %0 = add nsw i64 %min, 1
  %1 = bitcast %_elem_type_of_ap* %ap_rvo_based_addr_15 to i8*
  %uglygep = getelementptr i8, i8* %1, i64 4
  %2 = bitcast %_elem_type_of_y* %y_rvo_based_addr_13 to i8*
  %uglygep4 = getelementptr i8, i8* %2, i64 4
  br label %_loop_1_do_

_loop_1_do_:                                      ; preds = %_scf_2_skip_, %_loop_1_do_.lr.ph
  %indvar = phi i64 [ %3, %_scf_2_skip_ ], [ 0, %_loop_1_do_.lr.ph ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %_scf_2_skip_ ], [ 2, %_loop_1_do_.lr.ph ]
  %appos.023 = phi i64 [ %_add_tmp56, %_scf_2_skip_ ], [ 0, %_loop_1_do_.lr.ph ]
  %jj.021 = phi i64 [ %_loop_1_update_loop_ix, %_scf_2_skip_ ], [ 1, %_loop_1_do_.lr.ph ]
  %3 = add i64 %indvar, 1
  %_ind_cast = getelementptr %_elem_type_of_x, %_elem_type_of_x* %x_rvo_based_addr_11, i64 %jj.021, i32 0
  %_val_x_ = load float, float* %_ind_cast, align 4
  %_equ_tmp = fcmp contract une float %_val_x_, 0.000000e+00
  br i1 %_equ_tmp, label %_scf_2_true, label %_scf_2_continue_

_scf_2_continue_:                                 ; preds = %_loop_1_do_
  %_ind_cast8 = getelementptr %_elem_type_of_y, %_elem_type_of_y* %y_rvo_based_addr_13, i64 %jj.021, i32 0
  %_val_y_ = load float, float* %_ind_cast8, align 4
  %_equ_tmp10 = fcmp contract une float %_val_y_, 0.000000e+00
  br i1 %_equ_tmp10, label %_scf_2_true, label %_scf_2_skip_

_scf_2_true:                                      ; preds = %_scf_2_continue_, %_loop_1_do_
  %_val_alpha_27 = load float, float* %.alpha, align 4
  %_mult_tmp25 = fmul contract float %_val_x_, %_val_alpha_27
  %min.iters.check = icmp ult i64 %3, 32
  br i1 %min.iters.check, label %_loop_3_do_.preheader, label %vector.ph

_loop_3_do_.preheader:                            ; preds = %middle.block, %_scf_2_true
  %ii.019.ph = phi i64 [ 1, %_scf_2_true ], [ %ind.end, %middle.block ]
  br label %_loop_3_do_

vector.ph:                                        ; preds = %_scf_2_true
  %n.vec = and i64 %3, -32
  %ind.end = or i64 %n.vec, 1
  %broadcast.splatinsert51 = insertelement <4 x float> undef, float %_mult_tmp25, i32 0
  %broadcast.splat52 = shufflevector <4 x float> %broadcast.splatinsert51, <4 x float> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %4 = add i64 %index, %appos.023
  %5 = shl i64 %4, 2
  %uglygep2 = getelementptr i8, i8* %uglygep, i64 %5
  %6 = bitcast i8* %uglygep2 to float*
  %7 = bitcast float* %6 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %7, align 4
  %8 = getelementptr float, float* %6, i64 4
  %9 = bitcast float* %8 to <4 x float>*
  %wide.load36 = load <4 x float>, <4 x float>* %9, align 4
  %10 = getelementptr float, float* %6, i64 8
  %11 = bitcast float* %10 to <4 x float>*
  %wide.load37 = load <4 x float>, <4 x float>* %11, align 4
  %12 = getelementptr float, float* %6, i64 12
  %13 = bitcast float* %12 to <4 x float>*
  %wide.load38 = load <4 x float>, <4 x float>* %13, align 4
  %14 = getelementptr float, float* %6, i64 16
  %15 = bitcast float* %14 to <4 x float>*
  %wide.load39 = load <4 x float>, <4 x float>* %15, align 4
  %16 = getelementptr float, float* %6, i64 20
  %17 = bitcast float* %16 to <4 x float>*
  %wide.load40 = load <4 x float>, <4 x float>* %17, align 4
  %18 = getelementptr float, float* %6, i64 24
  %19 = bitcast float* %18 to <4 x float>*
  %wide.load41 = load <4 x float>, <4 x float>* %19, align 4
  %20 = getelementptr float, float* %6, i64 28
  %21 = bitcast float* %20 to <4 x float>*
  %wide.load42 = load <4 x float>, <4 x float>* %21, align 4
  %22 = shl i64 %index, 2
  %uglygep5 = getelementptr i8, i8* %uglygep4, i64 %22
  %23 = bitcast i8* %uglygep5 to float*
  %24 = bitcast float* %23 to <4 x float>*
  %wide.load43 = load <4 x float>, <4 x float>* %24, align 4
  %25 = getelementptr float, float* %23, i64 4
  %26 = bitcast float* %25 to <4 x float>*
  %wide.load44 = load <4 x float>, <4 x float>* %26, align 4
  %27 = getelementptr float, float* %23, i64 8
  %28 = bitcast float* %27 to <4 x float>*
  %wide.load45 = load <4 x float>, <4 x float>* %28, align 4
  %29 = getelementptr float, float* %23, i64 12
  %30 = bitcast float* %29 to <4 x float>*
  %wide.load46 = load <4 x float>, <4 x float>* %30, align 4
  %31 = getelementptr float, float* %23, i64 16
  %32 = bitcast float* %31 to <4 x float>*
  %wide.load47 = load <4 x float>, <4 x float>* %32, align 4
  %33 = getelementptr float, float* %23, i64 20
  %34 = bitcast float* %33 to <4 x float>*
  %wide.load48 = load <4 x float>, <4 x float>* %34, align 4
  %35 = getelementptr float, float* %23, i64 24
  %36 = bitcast float* %35 to <4 x float>*
  %wide.load49 = load <4 x float>, <4 x float>* %36, align 4
  %37 = getelementptr float, float* %23, i64 28
  %38 = bitcast float* %37 to <4 x float>*
  %wide.load50 = load <4 x float>, <4 x float>* %38, align 4
  %39 = fmul contract <4 x float> %broadcast.splat52, %wide.load43
  %40 = fmul contract <4 x float> %broadcast.splat52, %wide.load44
  %41 = fmul contract <4 x float> %broadcast.splat52, %wide.load45
  %42 = fmul contract <4 x float> %broadcast.splat52, %wide.load46
  %43 = fmul contract <4 x float> %broadcast.splat52, %wide.load47
  %44 = fmul contract <4 x float> %broadcast.splat52, %wide.load48
  %45 = fmul contract <4 x float> %broadcast.splat52, %wide.load49
  %46 = fmul contract <4 x float> %broadcast.splat52, %wide.load50
  %47 = fadd contract <4 x float> %wide.load, %39
  %48 = fadd contract <4 x float> %wide.load36, %40
  %49 = fadd contract <4 x float> %wide.load37, %41
  %50 = fadd contract <4 x float> %wide.load38, %42
  %51 = fadd contract <4 x float> %wide.load39, %43
  %52 = fadd contract <4 x float> %wide.load40, %44
  %53 = fadd contract <4 x float> %wide.load41, %45
  %54 = fadd contract <4 x float> %wide.load42, %46
  store <4 x float> %47, <4 x float>* %7, align 4
  store <4 x float> %48, <4 x float>* %9, align 4
  store <4 x float> %49, <4 x float>* %11, align 4
  store <4 x float> %50, <4 x float>* %13, align 4
  store <4 x float> %51, <4 x float>* %15, align 4
  store <4 x float> %52, <4 x float>* %17, align 4
  store <4 x float> %53, <4 x float>* %19, align 4
  store <4 x float> %54, <4 x float>* %21, align 4
  %index.next = add i64 %index, 32
  %55 = icmp eq i64 %index.next, %n.vec
  br i1 %55, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %3, %n.vec
  br i1 %cmp.n, label %_scf_2_skip_, label %_loop_3_do_.preheader

_scf_2_skip_.loopexit:                            ; preds = %_loop_3_do_
  br label %_scf_2_skip_

_scf_2_skip_:                                     ; preds = %_scf_2_skip_.loopexit, %middle.block, %_scf_2_continue_
  %_sub_tmp = add i64 %_add_tmp55, %appos.023
  %_add_tmp56 = add i64 %_sub_tmp, %jj.021
  %_loop_1_update_loop_ix = add nuw nsw i64 %jj.021, 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond28 = icmp eq i64 %indvars.iv, %0
  br i1 %exitcond28, label %_return_bb.loopexit, label %_loop_1_do_

_loop_3_do_:                                      ; preds = %_loop_3_do_.preheader, %_loop_3_do_
  %ii.019 = phi i64 [ %_loop_3_update_loop_ix, %_loop_3_do_ ], [ %ii.019.ph, %_loop_3_do_.preheader ]
  %_add_tmp27 = add nsw i64 %ii.019, %appos.023
  %_ind_cast29 = getelementptr %_elem_type_of_ap, %_elem_type_of_ap* %ap_rvo_based_addr_15, i64 %_add_tmp27, i32 0
  %_val_ap_ = load float, float* %_ind_cast29, align 4
  %_ind_cast49 = getelementptr %_elem_type_of_y, %_elem_type_of_y* %y_rvo_based_addr_13, i64 %ii.019, i32 0
  %_val_y_50 = load float, float* %_ind_cast49, align 4
  %_mult_tmp51 = fmul contract float %_mult_tmp25, %_val_y_50
  %_add_tmp52 = fadd contract float %_val_ap_, %_mult_tmp51
  store float %_add_tmp52, float* %_ind_cast29, align 4
  %_loop_3_update_loop_ix = add nuw nsw i64 %ii.019, 1
  %exitcond = icmp eq i64 %_loop_3_update_loop_ix, %indvars.iv
  br i1 %exitcond, label %_scf_2_skip_.loopexit, label %_loop_3_do_

_return_bb.loopexit:                              ; preds = %_scf_2_skip_
  br label %_return_bb

_return_bb:                                       ; preds = %_return_bb.loopexit, %entry
  ret void
}
