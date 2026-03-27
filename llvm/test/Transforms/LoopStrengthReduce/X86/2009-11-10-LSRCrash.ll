; RUN: llc < %s -mtriple=i386-apple-darwin11

define void @_ZN4llvm20SelectionDAGLowering14visitInlineAsmENS_8CallSiteE(i1 %arg) nounwind ssp align 2 {
entry:
  br i1 %arg, label %bb3.i, label %bb4.i

bb3.i:                                            ; preds = %entry
  unreachable

bb4.i:                                            ; preds = %entry
  br i1 %arg, label %bb.i.i, label %_ZNK4llvm8CallSite14getCalledValueEv.exit

bb.i.i:                                           ; preds = %bb4.i
  unreachable

_ZNK4llvm8CallSite14getCalledValueEv.exit:        ; preds = %bb4.i
  br i1 %arg, label %_ZN4llvm4castINS_9InlineAsmEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit, label %bb6.i

bb6.i:                                            ; preds = %_ZNK4llvm8CallSite14getCalledValueEv.exit
  unreachable

_ZN4llvm4castINS_9InlineAsmEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit: ; preds = %_ZNK4llvm8CallSite14getCalledValueEv.exit
  br i1 %arg, label %_ZL25hasInlineAsmMemConstraintRSt6vectorIN4llvm9InlineAsm14ConstraintInfoESaIS2_EERKNS0_14TargetLoweringE.exit, label %bb.i

bb.i:                                             ; preds = %_ZN4llvm4castINS_9InlineAsmEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit
  br label %_ZL25hasInlineAsmMemConstraintRSt6vectorIN4llvm9InlineAsm14ConstraintInfoESaIS2_EERKNS0_14TargetLoweringE.exit

_ZL25hasInlineAsmMemConstraintRSt6vectorIN4llvm9InlineAsm14ConstraintInfoESaIS2_EERKNS0_14TargetLoweringE.exit: ; preds = %bb.i, %_ZN4llvm4castINS_9InlineAsmEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit
  br i1 %arg, label %bb50, label %bb27

bb27:                                             ; preds = %_ZL25hasInlineAsmMemConstraintRSt6vectorIN4llvm9InlineAsm14ConstraintInfoESaIS2_EERKNS0_14TargetLoweringE.exit
  br i1 %arg, label %bb1.i727, label %bb.i.i726

bb.i.i726:                                        ; preds = %bb27
  unreachable

bb1.i727:                                         ; preds = %bb27
  unreachable

bb50:                                             ; preds = %_ZL25hasInlineAsmMemConstraintRSt6vectorIN4llvm9InlineAsm14ConstraintInfoESaIS2_EERKNS0_14TargetLoweringE.exit
  br label %bb107

bb51:                                             ; preds = %bb107
  br i1 %arg, label %bb105, label %bb106

bb105:                                            ; preds = %bb51
  unreachable

bb106:                                            ; preds = %bb51
  br label %bb107

bb107:                                            ; preds = %bb106, %bb50
  br i1 %arg, label %bb108, label %bb51

bb108:                                            ; preds = %bb107
  br i1 %arg, label %bb242, label %bb114

bb114:                                            ; preds = %bb108
  br i1 %arg, label %bb141, label %bb116

bb116:                                            ; preds = %bb114
  br i1 %arg, label %bb120, label %bb121

bb120:                                            ; preds = %bb116
  unreachable

bb121:                                            ; preds = %bb116
  unreachable

bb141:                                            ; preds = %bb114
  br i1 %arg, label %bb182, label %bb143

bb143:                                            ; preds = %bb141
  br label %bb157

bb144:                                            ; preds = %bb.i.i.i843
  switch i32 undef, label %bb155 [
    i32 2, label %bb153
    i32 6, label %bb153
    i32 4, label %bb153
  ]

bb153:                                            ; preds = %bb144, %bb144, %bb144
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=1]
  br label %bb157

bb155:                                            ; preds = %bb144
  unreachable

bb157:                                            ; preds = %bb153, %bb143
  %indvar = phi i32 [ %indvar.next, %bb153 ], [ 0, %bb143 ] ; <i32> [#uses=2]
  %0 = icmp eq i32 undef, %indvar                 ; <i1> [#uses=1]
  switch i16 undef, label %bb6.i841 [
    i16 9, label %_ZN4llvm4castINS_14ConstantSDNodeENS_7SDValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS5_.exit
    i16 26, label %_ZN4llvm4castINS_14ConstantSDNodeENS_7SDValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS5_.exit
  ]

bb6.i841:                                         ; preds = %bb157
  unreachable

_ZN4llvm4castINS_14ConstantSDNodeENS_7SDValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS5_.exit: ; preds = %bb157, %bb157
  br i1 %arg, label %bb.i.i.i843, label %bb1.i.i.i844

bb.i.i.i843:                                      ; preds = %_ZN4llvm4castINS_14ConstantSDNodeENS_7SDValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS5_.exit
  br i1 %0, label %bb158, label %bb144

bb1.i.i.i844:                                     ; preds = %_ZN4llvm4castINS_14ConstantSDNodeENS_7SDValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS5_.exit
  unreachable

bb158:                                            ; preds = %bb.i.i.i843
  br i1 %arg, label %bb177, label %bb176

bb176:                                            ; preds = %bb158
  unreachable

bb177:                                            ; preds = %bb158
  br i1 %arg, label %bb179, label %bb178

bb178:                                            ; preds = %bb177
  unreachable

bb179:                                            ; preds = %bb177
  unreachable

bb182:                                            ; preds = %bb141
  unreachable

bb242:                                            ; preds = %bb108
  unreachable
}
