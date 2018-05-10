; RUN: opt < %s -analyze -iv-users
; This is a regression test against very slow execution...
; In bad case it should fail by timeout.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

define void @quux(i8 addrspace(1)* %arg, i8 addrspace(1)* %arg1) {
bb:
  %tmp2 = getelementptr inbounds i8, i8 addrspace(1)* %arg, i64 80
  %tmp3 = bitcast i8 addrspace(1)* %tmp2 to i8 addrspace(1)* addrspace(1)*
  %tmp4 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %tmp3, align 8
  %tmp5 = getelementptr inbounds i8, i8 addrspace(1)* %tmp4, i64 8
  %tmp6 = bitcast i8 addrspace(1)* %tmp5 to i8 addrspace(1)* addrspace(1)*
  %tmp7 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %tmp6, align 8
  %tmp8 = getelementptr inbounds i8, i8 addrspace(1)* %tmp7, i64 8
  %tmp9 = bitcast i8 addrspace(1)* %tmp8 to i32 addrspace(1)*
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 8
  %tmp11 = udiv i32 65, %tmp10
  %tmp12 = getelementptr inbounds i8, i8 addrspace(1)* %arg, i64 80
  %tmp13 = bitcast i8 addrspace(1)* %tmp12 to i8 addrspace(1)* addrspace(1)*
  %tmp14 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %tmp13, align 8
  %tmp15 = getelementptr inbounds i8, i8 addrspace(1)* %tmp14, i64 8
  %tmp16 = bitcast i8 addrspace(1)* %tmp15 to i8 addrspace(1)* addrspace(1)*
  %tmp17 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %tmp16, align 8
  %tmp18 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 8
  %tmp19 = bitcast i8 addrspace(1)* %tmp18 to i32 addrspace(1)*
  %tmp20 = load i32, i32 addrspace(1)* %tmp19, align 8, !range !0
  %tmp21 = getelementptr inbounds i8, i8 addrspace(1)* %tmp17, i64 8
  %tmp22 = bitcast i8 addrspace(1)* %tmp21 to i32 addrspace(1)*
  %tmp23 = load i32, i32 addrspace(1)* %tmp22, align 8, !range !0
  %tmp24 = zext i32 %tmp23 to i64
  %tmp25 = and i32 %tmp11, 7
  %tmp26 = icmp ugt i32 %tmp10, 9
  br i1 %tmp26, label %bb27, label %bb46

bb27:                                             ; preds = %bb117, %bb
  %tmp28 = phi i32 [ 8, %bb ], [ %tmp112, %bb117 ]
  br label %bb29

bb29:                                             ; preds = %bb40, %bb27
  %tmp30 = phi i32 [ %tmp43, %bb40 ], [ %tmp28, %bb27 ]
  %tmp31 = phi i32 [ %tmp41, %bb40 ], [ %tmp25, %bb27 ]
  br label %bb32

bb32:                                             ; preds = %bb37, %bb29
  %tmp33 = phi i64 [ 0, %bb29 ], [ %tmp38, %bb37 ]
  %tmp34 = trunc i64 %tmp33 to i32
  %tmp35 = add i32 %tmp30, %tmp34
  %tmp36 = icmp ult i32 %tmp35, %tmp20
  br i1 %tmp36, label %bb37, label %bb56

bb37:                                             ; preds = %bb32
  %tmp38 = add nuw nsw i64 %tmp33, 1
  %tmp39 = icmp ult i64 %tmp38, %tmp24
  br i1 %tmp39, label %bb32, label %bb40

bb40:                                             ; preds = %bb37
  %tmp41 = add i32 %tmp31, -1
  %tmp42 = trunc i64 %tmp38 to i32
  %tmp43 = add i32 %tmp30, %tmp42
  %tmp44 = icmp eq i32 %tmp41, 0
  br i1 %tmp44, label %bb45, label %bb29

bb45:                                             ; preds = %bb40
  ret void

bb46:                                             ; preds = %bb
  %tmp47 = sub nsw i32 %tmp11, %tmp25
  br label %bb48

bb48:                                             ; preds = %bb117, %bb46
  %tmp49 = phi i32 [ 8, %bb46 ], [ %tmp112, %bb117 ]
  %tmp50 = phi i32 [ %tmp47, %bb46 ], [ %tmp118, %bb117 ]
  br label %bb51

bb51:                                             ; preds = %bb58, %bb48
  %tmp52 = phi i64 [ 0, %bb48 ], [ %tmp59, %bb58 ]
  %tmp53 = phi i32 [ %tmp49, %bb48 ], [ %tmp54, %bb58 ]
  %tmp54 = add i32 %tmp53, 1
  %tmp55 = icmp ult i32 %tmp53, %tmp20
  br i1 %tmp55, label %bb58, label %bb56

bb56:                                             ; preds = %bb109, %bb101, %bb93, %bb85, %bb77, %bb69, %bb61, %bb51, %bb32
  unreachable

bb58:                                             ; preds = %bb51
  %tmp59 = add nuw nsw i64 %tmp52, 1
  %tmp60 = icmp ult i64 %tmp59, %tmp24
  br i1 %tmp60, label %bb51, label %bb61

bb61:                                             ; preds = %bb66, %bb58
  %tmp62 = phi i64 [ %tmp67, %bb66 ], [ 0, %bb58 ]
  %tmp63 = phi i32 [ %tmp64, %bb66 ], [ %tmp54, %bb58 ]
  %tmp64 = add i32 %tmp63, 1
  %tmp65 = icmp ult i32 %tmp63, %tmp20
  br i1 %tmp65, label %bb66, label %bb56

bb66:                                             ; preds = %bb61
  %tmp67 = add nuw nsw i64 %tmp62, 1
  %tmp68 = icmp ult i64 %tmp67, %tmp24
  br i1 %tmp68, label %bb61, label %bb69

bb69:                                             ; preds = %bb74, %bb66
  %tmp70 = phi i64 [ %tmp75, %bb74 ], [ 0, %bb66 ]
  %tmp71 = phi i32 [ %tmp72, %bb74 ], [ %tmp64, %bb66 ]
  %tmp72 = add i32 %tmp71, 1
  %tmp73 = icmp ult i32 %tmp71, %tmp20
  br i1 %tmp73, label %bb74, label %bb56

bb74:                                             ; preds = %bb69
  %tmp75 = add nuw nsw i64 %tmp70, 1
  %tmp76 = icmp ult i64 %tmp75, %tmp24
  br i1 %tmp76, label %bb69, label %bb77

bb77:                                             ; preds = %bb82, %bb74
  %tmp78 = phi i64 [ %tmp83, %bb82 ], [ 0, %bb74 ]
  %tmp79 = phi i32 [ %tmp80, %bb82 ], [ %tmp72, %bb74 ]
  %tmp80 = add i32 %tmp79, 1
  %tmp81 = icmp ult i32 %tmp79, %tmp20
  br i1 %tmp81, label %bb82, label %bb56

bb82:                                             ; preds = %bb77
  %tmp83 = add nuw nsw i64 %tmp78, 1
  %tmp84 = icmp ult i64 %tmp83, %tmp24
  br i1 %tmp84, label %bb77, label %bb85

bb85:                                             ; preds = %bb90, %bb82
  %tmp86 = phi i64 [ %tmp91, %bb90 ], [ 0, %bb82 ]
  %tmp87 = phi i32 [ %tmp88, %bb90 ], [ %tmp80, %bb82 ]
  %tmp88 = add i32 %tmp87, 1
  %tmp89 = icmp ult i32 %tmp87, %tmp20
  br i1 %tmp89, label %bb90, label %bb56

bb90:                                             ; preds = %bb85
  %tmp91 = add nuw nsw i64 %tmp86, 1
  %tmp92 = icmp ult i64 %tmp91, %tmp24
  br i1 %tmp92, label %bb85, label %bb93

bb93:                                             ; preds = %bb98, %bb90
  %tmp94 = phi i64 [ %tmp99, %bb98 ], [ 0, %bb90 ]
  %tmp95 = phi i32 [ %tmp96, %bb98 ], [ %tmp88, %bb90 ]
  %tmp96 = add i32 %tmp95, 1
  %tmp97 = icmp ult i32 %tmp95, %tmp20
  br i1 %tmp97, label %bb98, label %bb56

bb98:                                             ; preds = %bb93
  %tmp99 = add nuw nsw i64 %tmp94, 1
  %tmp100 = icmp ult i64 %tmp99, %tmp24
  br i1 %tmp100, label %bb93, label %bb101

bb101:                                            ; preds = %bb106, %bb98
  %tmp102 = phi i64 [ %tmp107, %bb106 ], [ 0, %bb98 ]
  %tmp103 = phi i32 [ %tmp104, %bb106 ], [ %tmp96, %bb98 ]
  %tmp104 = add i32 %tmp103, 1
  %tmp105 = icmp ult i32 %tmp103, %tmp20
  br i1 %tmp105, label %bb106, label %bb56

bb106:                                            ; preds = %bb101
  %tmp107 = add nuw nsw i64 %tmp102, 1
  %tmp108 = icmp ult i64 %tmp107, %tmp24
  br i1 %tmp108, label %bb101, label %bb109

bb109:                                            ; preds = %bb114, %bb106
  %tmp110 = phi i64 [ %tmp115, %bb114 ], [ 0, %bb106 ]
  %tmp111 = phi i32 [ %tmp112, %bb114 ], [ %tmp104, %bb106 ]
  %tmp112 = add i32 %tmp111, 1
  %tmp113 = icmp ult i32 %tmp111, %tmp20
  br i1 %tmp113, label %bb114, label %bb56

bb114:                                            ; preds = %bb109
  %tmp115 = add nuw nsw i64 %tmp110, 1
  %tmp116 = icmp ult i64 %tmp115, %tmp24
  br i1 %tmp116, label %bb109, label %bb117

bb117:                                            ; preds = %bb114
  %tmp118 = add i32 %tmp50, -8
  %tmp119 = icmp eq i32 %tmp118, 0
  br i1 %tmp119, label %bb27, label %bb48
}

!0 = !{i32 0, i32 2147483647}
