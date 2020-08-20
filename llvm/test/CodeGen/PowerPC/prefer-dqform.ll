; RUN: llc -verify-machineinstrs -disable-ppc-instr-form-prep=true -mcpu=pwr9 < %s \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names | FileCheck %s -check-prefix=CHECK-P9
; RUN: llc -verify-machineinstrs -disable-ppc-instr-form-prep=true -mcpu=pwr10 < %s \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names | FileCheck %s -check-prefix=CHECK-P10

target triple = "powerpc64le-unknown-linux-gnu"

%_elem_type_of_a = type <{ double }>
%_elem_type_of_x = type <{ double }>
%_elem_type_of_y = type <{ double }>

define void @test(i32* dereferenceable(4) %.ial, i32* noalias dereferenceable(4) %.m, i32* noalias dereferenceable(4) %.n, [0 x %_elem_type_of_a]*  %.a, i32* noalias dereferenceable(4) %.lda, [0 x %_elem_type_of_x]* noalias %.x, [0 x %_elem_type_of_y]* noalias %.y) {
; CHECK-P9-LABEL: test:
; CHECK-P9:       .LBB0_2: # %_loop_2_do_
; CHECK-P9:         lxv vs1, -16(r4)
; CHECK-P9:         lxv vs2, 0(r4)
; CHECK-P9-DAG:     lxv vs3, -16(r3)
; CHECK-P9-DAG:     lxv vs4, 0(r3)
; CHECK-P9-DAG:     xvmaddadp vs1, vs3, vs1
; CHECK-P9-DAG:     stxv vs1, -16(r4)
; CHECK-P9-DAG:     xvmaddadp vs2, vs4, vs0
; CHECK-P9:         stxv vs2, 0(r4)
; CHECK-P9:         bdnz .LBB0_2
;
; FIXME: use pair load/store instructions lxvp/stxvp
; CHECK-P10-LABEL: test:
; CHECK-P10:       .LBB0_2: # %_loop_2_do_
; CHECK-P10:         lxv vs1, -16(r4)
; CHECK-P10:         lxv vs2, 0(r4)
; CHECK-P10-DAG:     lxv vs3, -16(r3)
; CHECK-P10-DAG:     lxv vs4, 0(r3)
; CHECK-P10-DAG:     xvmaddadp vs1, vs3, vs1
; CHECK-P10-DAG:     xvmaddadp vs2, vs4, vs0
; CHECK-P10-DAG:     stxv vs1, -16(r4)
; CHECK-P10:         stxv vs2, 0(r4)
; CHECK-P10:         bdnz .LBB0_2
test_entry:
  %_conv5 = ptrtoint [0 x %_elem_type_of_a]* %.a to i64
  %_andi_tmp = and i64 %_conv5, 15
  %_equ_tmp = icmp eq i64 %_andi_tmp, 0
  %. = select i1 %_equ_tmp, i32 1, i32 2
  %_val_m_ = load i32, i32* %.m, align 4
  %_sub_tmp9 = sub nsw i32 1, %.
  %_add_tmp10 = add i32 %_sub_tmp9, %_val_m_
  %_mod_tmp = srem i32 %_add_tmp10, 16
  %_sub_tmp11 = sub i32 %_val_m_, %_mod_tmp
  %_val_n_ = load i32, i32* %.n, align 4
  %x_rvo_based_addr_17 = getelementptr inbounds [0 x %_elem_type_of_x], [0 x %_elem_type_of_x]* %.x, i64 0, i64 -1
  %_div_tmp = sdiv i32 %_val_n_, 2
  %_conv16 = sext i32 %_div_tmp to i64
  %_ind_cast = getelementptr inbounds %_elem_type_of_x, %_elem_type_of_x* %x_rvo_based_addr_17, i64 %_conv16, i32 0
  %_val_x_ = load double, double* %_ind_cast, align 8
  %.splatinsert = insertelement <2 x double> undef, double %_val_x_, i32 0
  %.splat = shufflevector <2 x double> %.splatinsert, <2 x double> undef, <2 x i32> zeroinitializer
  %_grt_tmp21 = icmp sgt i32 %., %_sub_tmp11
  br i1 %_grt_tmp21, label %_return_bb, label %_loop_2_do_.lr.ph

_loop_2_do_.lr.ph:                                ; preds = %test_entry
  %_val_lda_ = load i32, i32* %.lda, align 4
  %_conv = sext i32 %_val_lda_ to i64
  %_mult_tmp = shl nsw i64 %_conv, 3
  %_sub_tmp4 = sub nuw nsw i64 -8, %_mult_tmp
  %y_rvo_based_addr_19 = getelementptr inbounds [0 x %_elem_type_of_y], [0 x %_elem_type_of_y]* %.y, i64 0, i64 -1
  %a_byte_ptr_ = bitcast [0 x %_elem_type_of_a]* %.a to i8*
  %a_rvo_based_addr_ = getelementptr inbounds i8, i8* %a_byte_ptr_, i64 %_sub_tmp4
  %0 = zext i32 %. to i64
  %1 = sext i32 %_sub_tmp11 to i64
  br label %_loop_2_do_

_loop_2_do_:                                      ; preds = %_loop_2_do_.lr.ph, %_loop_2_do_
  %indvars.iv = phi i64 [ %0, %_loop_2_do_.lr.ph ], [ %indvars.iv.next, %_loop_2_do_ ]
  %_ix_x_len19 = shl nuw nsw i64 %indvars.iv, 3
  %y_ix_dim_0_20 = getelementptr inbounds %_elem_type_of_y, %_elem_type_of_y* %y_rvo_based_addr_19, i64 %indvars.iv
  %2 = bitcast %_elem_type_of_y* %y_ix_dim_0_20 to <2 x double>*
  %3 = load <2 x double>, <2 x double>* %2, align 1
  %4 = getelementptr %_elem_type_of_y, %_elem_type_of_y* %y_ix_dim_0_20, i64 2
  %5 = bitcast %_elem_type_of_y* %4 to <2 x double>*
  %6 = load <2 x double>, <2 x double>* %5, align 1
  %a_ix_dim_1_ = getelementptr inbounds i8, i8* %a_rvo_based_addr_, i64 %_ix_x_len19
  %7 = bitcast i8* %a_ix_dim_1_ to <2 x double>*
  %8 = load <2 x double>, <2 x double>* %7, align 1
  %9 = getelementptr i8, i8* %a_ix_dim_1_, i64 16
  %10 = bitcast i8* %9 to <2 x double>*
  %11 = load <2 x double>, <2 x double>* %10, align 1
  %12 = tail call nsz contract <2 x double> @llvm.fma.v2f64(<2 x double> %8, <2 x double> %3, <2 x double> %3)
  %13 = tail call nsz contract <2 x double> @llvm.fma.v2f64(<2 x double> %11, <2 x double> %.splat, <2 x double> %6)
  store <2 x double> %12, <2 x double>* %2, align 1
  store <2 x double> %13, <2 x double>* %5, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 16
  %_grt_tmp = icmp sgt i64 %indvars.iv.next, %1
  br i1 %_grt_tmp, label %_return_bb, label %_loop_2_do_

_return_bb:                                       ; preds = %_loop_2_do_, %test_entry
  ret void
}

declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)
