; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; void matvecmul(const double *__restrict y, const double * __restrict x, const double * __restrict b,
;    const int * __restrict nb, const int * __restrict nx, const int * __restrict ny, const int * __restrict nz) {
;
;   for (int k=1;k<nz,++k) 
;      for (int j=1;j<ny,++j)
;        for (int i=1;i<nx,++i)
;          for (int l=1;l<nb,++l)
;            for (int m=1;m<nb,++m)
;                 y[k+1][j][i][l] = y[k+1][j][i][l] + b[k][j][i][m][l]*x[k][j][i][m]
; }

; CHECK-DAG: Loop 'k_loop' has cost = 30000000000
; CHECK-DAG: Loop 'j_loop' has cost = 30000000000
; CHECK-DAG: Loop 'i_loop' has cost = 30000000000
; CHECK-DAG: Loop 'm_loop' has cost = 10700000000
; CHECK-DAG: Loop 'l_loop' has cost = 1300000000

%_elem_type_of_double = type <{ double }>

; Function Attrs: norecurse nounwind
define void @mat_vec_mpy([0 x %_elem_type_of_double]* noalias %y, [0 x %_elem_type_of_double]* noalias readonly %x,
    [0 x %_elem_type_of_double]* noalias readonly %b, i32* noalias readonly %nb, i32* noalias readonly %nx, 
    i32* noalias readonly %ny, i32* noalias readonly %nz) {
mat_times_vec_entry:
  %_ind_val = load i32, i32* %nb, align 4
  %_conv = sext i32 %_ind_val to i64
  %_grt_tmp.i = icmp sgt i64 %_conv, 0
  %a_b.i = select i1 %_grt_tmp.i, i64 %_conv, i64 0
  %_ind_val1 = load i32, i32* %nx, align 4
  %_conv2 = sext i32 %_ind_val1 to i64
  %_grt_tmp.i266 = icmp sgt i64 %_conv2, 0
  %a_b.i267 = select i1 %_grt_tmp.i266, i64 %_conv2, i64 0
  %_ind_val3 = load i32, i32* %ny, align 4
  %_conv4 = sext i32 %_ind_val3 to i64
  %_grt_tmp.i264 = icmp sgt i64 %_conv4, 0
  %a_b.i265 = select i1 %_grt_tmp.i264, i64 %_conv4, i64 0
  %_ind_val5 = load i32, i32* %nz, align 4
  %_mult_tmp = shl nsw i64 %a_b.i, 3
  %_mult_tmp7 = mul i64 %_mult_tmp, %a_b.i267
  %_mult_tmp8 = mul i64 %_mult_tmp7, %a_b.i265
  %_sub_tmp = sub nuw nsw i64 -8, %_mult_tmp
  %_sub_tmp21 = sub i64 %_sub_tmp, %_mult_tmp7
  %_sub_tmp23 = sub i64 %_sub_tmp21, %_mult_tmp8
  %_mult_tmp73 = mul i64 %_mult_tmp, %a_b.i
  %_mult_tmp74 = mul i64 %_mult_tmp73, %a_b.i267
  %_mult_tmp75 = mul i64 %_mult_tmp74, %a_b.i265
  %_sub_tmp93 = sub i64 %_sub_tmp, %_mult_tmp73
  %_sub_tmp95 = sub i64 %_sub_tmp93, %_mult_tmp74
  %_sub_tmp97 = sub i64 %_sub_tmp95, %_mult_tmp75
  %_grt_tmp853288 = icmp slt i32 %_ind_val5, 1
  br i1 %_grt_tmp853288, label %_return_bb, label %k_loop.lr.ph

k_loop.lr.ph:                                     ; preds = %mat_times_vec_entry
  %_grt_tmp851279 = icmp slt i32 %_ind_val3, 1
  %_grt_tmp847270 = icmp slt i32 %_ind_val, 1
  %_aa_conv = bitcast [0 x %_elem_type_of_double]* %y to i8*
  %_adda_ = getelementptr inbounds i8, i8* %_aa_conv, i64 %_sub_tmp23
  %_aa_conv434 = bitcast [0 x %_elem_type_of_double]* %x to i8*
  %_adda_435 = getelementptr inbounds i8, i8* %_aa_conv434, i64 %_sub_tmp23
  %_aa_conv785 = bitcast [0 x %_elem_type_of_double]* %b to i8*
  %_adda_786 = getelementptr inbounds i8, i8* %_aa_conv785, i64 %_sub_tmp97
  br i1 %_grt_tmp851279, label %k_loop.us.preheader, label %k_loop.lr.ph.split

k_loop.us.preheader:                              ; preds = %k_loop.lr.ph
  br label %_return_bb.loopexit

k_loop.lr.ph.split:                               ; preds = %k_loop.lr.ph
  %_grt_tmp849273 = icmp slt i32 %_ind_val1, 1
  br i1 %_grt_tmp849273, label %k_loop.us291.preheader, label %k_loop.lr.ph.split.split

k_loop.us291.preheader:                           ; preds = %k_loop.lr.ph.split
  br label %_return_bb.loopexit300

k_loop.lr.ph.split.split:                         ; preds = %k_loop.lr.ph.split
  br i1 %_grt_tmp847270, label %k_loop.us294.preheader, label %k_loop.preheader

k_loop.preheader:                                 ; preds = %k_loop.lr.ph.split.split
  %0 = add i32 %_ind_val, 1
  %1 = add i32 %_ind_val1, 1
  %2 = add i32 %_ind_val3, 1
  %3 = add i32 %_ind_val5, 1
  br label %k_loop

k_loop.us294.preheader:                           ; preds = %k_loop.lr.ph.split.split
  br label %_return_bb.loopexit301

k_loop:                                           ; preds = %k_loop._label_18_crit_edge.split.split.split, %k_loop.preheader
  %indvars.iv316 = phi i64 [ 1, %k_loop.preheader ], [ %indvars.iv.next317, %k_loop._label_18_crit_edge.split.split.split ]
  %indvars.iv.next317 = add nuw nsw i64 %indvars.iv316, 1
  %_ix_x_len = mul i64 %_mult_tmp8, %indvars.iv.next317
  %_ix_x_len410 = mul i64 %_mult_tmp75, %indvars.iv316
  %_ix_x_len822 = mul i64 %_mult_tmp8, %indvars.iv316
  br label %j_loop

j_loop:                                           ; preds = %j_loop._label_15_crit_edge.split.split, %k_loop
  %indvars.iv312 = phi i64 [ %indvars.iv.next313, %j_loop._label_15_crit_edge.split.split ], [ 1, %k_loop ]
  %_ix_x_len371 = mul i64 %_mult_tmp7, %indvars.iv312
  %_ix_x_len415 = mul i64 %_mult_tmp74, %indvars.iv312
  br label %i_loop

i_loop:                                           ; preds = %i_loop._label_12_crit_edge.split, %j_loop
  %indvars.iv307 = phi i64 [ %indvars.iv.next308, %i_loop._label_12_crit_edge.split ], [ 1, %j_loop ]
  %_ix_x_len375 = mul i64 %_mult_tmp, %indvars.iv307
  %_ix_x_len420 = mul i64 %_mult_tmp73, %indvars.iv307
  br label %l_loop

l_loop:                                           ; preds = %l_loop._label_9_crit_edge, %i_loop
  %indvars.iv303 = phi i64 [ %indvars.iv.next304, %l_loop._label_9_crit_edge ], [ 1, %i_loop ]
  %_ix_x_len378 = shl nuw nsw i64 %indvars.iv303, 3
  br label %m_loop

m_loop:                                           ; preds = %m_loop, %l_loop
  %indvars.iv = phi i64 [ %indvars.iv.next, %m_loop ], [ 1, %l_loop ]
  %_ix_x_len424 = mul i64 %_mult_tmp, %indvars.iv
  %_ix_x_len454 = shl nuw nsw i64 %indvars.iv, 3
  %_ixa_gep = getelementptr inbounds i8, i8* %_adda_, i64 %_ix_x_len
  %_ixa_gep791 = getelementptr inbounds i8, i8* %_adda_786, i64 %_ix_x_len410
  %_ixa_gep823 = getelementptr inbounds i8, i8* %_adda_435, i64 %_ix_x_len822
  %_ixa_gep372 = getelementptr inbounds i8, i8* %_ixa_gep, i64 %_ix_x_len371
  %_ixa_gep376 = getelementptr inbounds i8, i8* %_ixa_gep372, i64 %_ix_x_len375
  %_ixa_gep796 = getelementptr inbounds i8, i8* %_ixa_gep791, i64 %_ix_x_len415
  %_ixa_gep828 = getelementptr inbounds i8, i8* %_ixa_gep823, i64 %_ix_x_len371
  %_ixa_gep379 = getelementptr inbounds i8, i8* %_ixa_gep376, i64 %_ix_x_len378
  %_ixa_gep801 = getelementptr inbounds i8, i8* %_ixa_gep796, i64 %_ix_x_len420
  %_ixa_gep833 = getelementptr inbounds i8, i8* %_ixa_gep828, i64 %_ix_x_len375
  %_ixa_gep806 = getelementptr inbounds i8, i8* %_ixa_gep801, i64 %_ix_x_len378
  %_ixa_gep810 = getelementptr inbounds i8, i8* %_ixa_gep806, i64 %_ix_x_len424
  %_gepp = bitcast i8* %_ixa_gep379 to double*
  %_gepp813 = bitcast i8* %_ixa_gep810 to double*
  %_ind_val814 = load double, double* %_gepp813, align 8
  %_ixa_gep837 = getelementptr inbounds i8, i8* %_ixa_gep833, i64 %_ix_x_len454
  %_gepp840 = bitcast i8* %_ixa_gep837 to double*
  %_ind_val841 = load double, double* %_gepp840, align 8
  %_mult_tmp842 = fmul double %_ind_val814, %_ind_val841
  store double %_mult_tmp842, double* %_gepp, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %wide.trip.count = zext i32 %0 to i64
  %wide.trip.count305 = zext i32 %0 to i64
  %wide.trip.count309 = zext i32 %1 to i64
  %wide.trip.count314 = zext i32 %2 to i64
  %wide.trip.count319 = zext i32 %3 to i64
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %m_loop, label %l_loop._label_9_crit_edge

l_loop._label_9_crit_edge:                        ; preds = %m_loop
  %indvars.iv.next304 = add nuw nsw i64 %indvars.iv303, 1
  %exitcond306 = icmp ne i64 %indvars.iv.next304, %wide.trip.count305
  br i1 %exitcond306, label %l_loop, label %i_loop._label_12_crit_edge.split

i_loop._label_12_crit_edge.split:                 ; preds = %l_loop._label_9_crit_edge
  %indvars.iv.next308 = add nuw nsw i64 %indvars.iv307, 1
  %exitcond310 = icmp ne i64 %indvars.iv.next308, %wide.trip.count309
  br i1 %exitcond310, label %i_loop, label %j_loop._label_15_crit_edge.split.split

j_loop._label_15_crit_edge.split.split:           ; preds = %i_loop._label_12_crit_edge.split
  %indvars.iv.next313 = add nuw nsw i64 %indvars.iv312, 1
  %exitcond315 = icmp ne i64 %indvars.iv.next313, %wide.trip.count314
  br i1 %exitcond315, label %j_loop, label %k_loop._label_18_crit_edge.split.split.split

k_loop._label_18_crit_edge.split.split.split:     ; preds = %j_loop._label_15_crit_edge.split.split
  %exitcond320 = icmp ne i64 %indvars.iv.next317, %wide.trip.count319
  br i1 %exitcond320, label %k_loop, label %_return_bb.loopexit302

_return_bb.loopexit:                              ; preds = %k_loop.us.preheader
  br label %_return_bb

_return_bb.loopexit300:                           ; preds = %k_loop.us291.preheader
  br label %_return_bb

_return_bb.loopexit301:                           ; preds = %k_loop.us294.preheader
  br label %_return_bb

_return_bb.loopexit302:                           ; preds = %k_loop._label_18_crit_edge.split.split.split
  br label %_return_bb

_return_bb:                                       ; preds = %_return_bb.loopexit302, %_return_bb.loopexit301, %_return_bb.loopexit300, %_return_bb.loopexit, %mat_times_vec_entry
  ret void
}


