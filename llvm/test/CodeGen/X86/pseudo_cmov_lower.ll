; RUN: llc < %s -mtriple=i386-linux-gnu -o - | FileCheck %s 

; This test checks that only a single js gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo1:
; CHECK: js
; CHECK-NOT: js
define i32 @foo1(i32 %v1, i32 %v2, i32 %v3) nounwind {
entry:
  %cmp = icmp slt i32 %v1, 0
  %v2.v3 = select i1 %cmp, i32 %v2, i32 %v3
  %v1.v2 = select i1 %cmp, i32 %v1, i32 %v2
  %sub = sub i32 %v1.v2, %v2.v3
  ret i32 %sub
}

; This test checks that only a single js gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR. This makes
; sure the code for the lowering for opposite conditions gets tested.
; CHECK-LABEL: foo11:
; CHECK: js
; CHECK-NOT: js
; CHECK-NOT: jns
define i32 @foo11(i32 %v1, i32 %v2, i32 %v3) nounwind {
entry:
  %cmp1 = icmp slt i32 %v1, 0
  %v2.v3 = select i1 %cmp1, i32 %v2, i32 %v3
  %cmp2 = icmp sge i32 %v1, 0
  %v1.v2 = select i1 %cmp2, i32 %v1, i32 %v2
  %sub = sub i32 %v1.v2, %v2.v3
  ret i32 %sub
}

; This test checks that only a single js gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo2:
; CHECK: js
; CHECK-NOT: js
define i32 @foo2(i8 %v1, i8 %v2, i8 %v3) nounwind {
entry:
  %cmp = icmp slt i8 %v1, 0
  %v2.v3 = select i1 %cmp, i8 %v2, i8 %v3
  %v1.v2 = select i1 %cmp, i8 %v1, i8 %v2
  %t1 = sext i8 %v2.v3 to i32
  %t2 = sext i8 %v1.v2 to i32
  %sub = sub i32 %t1, %t2
  ret i32 %sub
}

; This test checks that only a single js gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo3:
; CHECK: js
; CHECK-NOT: js
define i32 @foo3(i16 %v1, i16 %v2, i16 %v3) nounwind {
entry:
  %cmp = icmp slt i16 %v1, 0
  %v2.v3 = select i1 %cmp, i16 %v2, i16 %v3
  %v1.v2 = select i1 %cmp, i16 %v1, i16 %v2
  %t1 = sext i16 %v2.v3 to i32
  %t2 = sext i16 %v1.v2 to i32
  %sub = sub i32 %t1, %t2
  ret i32 %sub
}

; This test checks that only a single js gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo4:
; CHECK: js
; CHECK-NOT: js
define float @foo4(i32 %v1, float %v2, float %v3, float %v4) nounwind {
entry:
  %cmp = icmp slt i32 %v1, 0
  %t1 = select i1 %cmp, float %v2, float %v3
  %t2 = select i1 %cmp, float %v3, float %v4
  %sub = fsub float %t1, %t2
  ret float %sub
}

; This test checks that only a single je gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo5:
; CHECK: je
; CHECK-NOT: je
define double @foo5(i32 %v1, double %v2, double %v3, double %v4) nounwind {
entry:
  %cmp = icmp eq i32 %v1, 0
  %t1 = select i1 %cmp, double %v2, double %v3
  %t2 = select i1 %cmp, double %v3, double %v4
  %sub = fsub double %t1, %t2
  ret double %sub
}

; This test checks that only a single je gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo6:
; CHECK: je
; CHECK-NOT: je
define <4 x float> @foo6(i32 %v1, <4 x float> %v2, <4 x float> %v3, <4 x float> %v4) nounwind {
entry:
  %cmp = icmp eq i32 %v1, 0
  %t1 = select i1 %cmp, <4 x float> %v2, <4 x float> %v3
  %t2 = select i1 %cmp, <4 x float> %v3, <4 x float> %v4
  %sub = fsub <4 x float> %t1, %t2
  ret <4 x float> %sub
}

; This test checks that only a single je gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo7:
; CHECK: je
; CHECK-NOT: je
define <2 x double> @foo7(i32 %v1, <2 x double> %v2, <2 x double> %v3, <2 x double> %v4) nounwind {
entry:
  %cmp = icmp eq i32 %v1, 0
  %t1 = select i1 %cmp, <2 x double> %v2, <2 x double> %v3
  %t2 = select i1 %cmp, <2 x double> %v3, <2 x double> %v4
  %sub = fsub <2 x double> %t1, %t2
  ret <2 x double> %sub
}

; This test checks that only a single ja gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR. This combines
; all the supported types together into one long string of selects based
; on the same condition.
; CHECK-LABEL: foo8:
; CHECK: ja
; CHECK-NOT: ja
define void @foo8(i32 %v1,
                  i8 %v2, i8 %v3,
                  i16 %v12, i16 %v13,
                  i32 %v22, i32 %v23,
                  float %v32, float %v33,
                  double %v42, double %v43,
                  <4 x float> %v52, <4 x float> %v53,
                  <2 x double> %v62, <2 x double> %v63,
                  <8 x float> %v72, <8 x float> %v73,
                  <4 x double> %v82, <4 x double> %v83,
                  <16 x float> %v92, <16 x float> %v93,
                  <8 x double> %v102, <8 x double> %v103,
                  i8 * %dst) nounwind {
entry:
  %add.ptr11 = getelementptr inbounds i8, i8* %dst, i32 2
  %a11 = bitcast i8* %add.ptr11 to i16*

  %add.ptr21 = getelementptr inbounds i8, i8* %dst, i32 4
  %a21 = bitcast i8* %add.ptr21 to i32*

  %add.ptr31 = getelementptr inbounds i8, i8* %dst, i32 8
  %a31 = bitcast i8* %add.ptr31 to float*

  %add.ptr41 = getelementptr inbounds i8, i8* %dst, i32 16
  %a41 = bitcast i8* %add.ptr41 to double*

  %add.ptr51 = getelementptr inbounds i8, i8* %dst, i32 32
  %a51 = bitcast i8* %add.ptr51 to <4 x float>*

  %add.ptr61 = getelementptr inbounds i8, i8* %dst, i32 48
  %a61 = bitcast i8* %add.ptr61 to <2 x double>*

  %add.ptr71 = getelementptr inbounds i8, i8* %dst, i32 64
  %a71 = bitcast i8* %add.ptr71 to <8 x float>*

  %add.ptr81 = getelementptr inbounds i8, i8* %dst, i32 128
  %a81 = bitcast i8* %add.ptr81 to <4 x double>*

  %add.ptr91 = getelementptr inbounds i8, i8* %dst, i32 64
  %a91 = bitcast i8* %add.ptr91 to <16 x float>*

  %add.ptr101 = getelementptr inbounds i8, i8* %dst, i32 128
  %a101 = bitcast i8* %add.ptr101 to <8 x double>*

  ; These operations are necessary, because select of two single use loads
  ; ends up getting optimized into a select of two leas, followed by a
  ; single load of the selected address.
  %t13 = xor i16 %v13, 11
  %t23 = xor i32 %v23, 1234
  %t33 = fadd float %v33, %v32
  %t43 = fadd double %v43, %v42
  %t53 = fadd <4 x float> %v53, %v52
  %t63 = fadd <2 x double> %v63, %v62
  %t73 = fsub <8 x float> %v73, %v72
  %t83 = fsub <4 x double> %v83, %v82
  %t93 = fsub <16 x float> %v93, %v92
  %t103 = fsub <8 x double> %v103, %v102

  %cmp = icmp ugt i32 %v1, 31
  %t11 = select i1 %cmp, i16 %v12, i16 %t13
  %t21 = select i1 %cmp, i32 %v22, i32 %t23
  %t31 = select i1 %cmp, float %v32, float %t33
  %t41 = select i1 %cmp, double %v42, double %t43
  %t51 = select i1 %cmp, <4 x float> %v52, <4 x float> %t53
  %t61 = select i1 %cmp, <2 x double> %v62, <2 x double> %t63
  %t71 = select i1 %cmp, <8 x float> %v72, <8 x float> %t73
  %t81 = select i1 %cmp, <4 x double> %v82, <4 x double> %t83
  %t91 = select i1 %cmp, <16 x float> %v92, <16 x float> %t93
  %t101 = select i1 %cmp, <8 x double> %v102, <8 x double> %t103

  store i16 %t11, i16* %a11, align 2
  store i32 %t21, i32* %a21, align 4
  store float %t31, float* %a31, align 4
  store double %t41, double* %a41, align 8
  store <4 x float> %t51, <4 x float>* %a51, align 16
  store <2 x double> %t61, <2 x double>* %a61, align 16
  store <8 x float> %t71, <8 x float>* %a71, align 32
  store <4 x double> %t81, <4 x double>* %a81, align 32
  store <16 x float> %t91, <16 x float>* %a91, align 32
  store <8 x double> %t101, <8 x double>* %a101, align 32

  ret void
}

; This test checks that only a single ja gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; on the same condition.
; Contrary to my expectations, this doesn't exercise the code for
; CMOV_V8I1, CMOV_V16I1, CMOV_V32I1, or CMOV_V64I1.  Instead the selects all
; get lowered into vector length number of selects, which all eventually turn
; into a huge number of CMOV_GR8, which are all contiguous, so the optimization
; kicks in as long as CMOV_GR8 is supported. I couldn't find a way to get
; CMOV_V*I1 pseudo-opcodes to get generated. If a way exists to get CMOV_V*1
; pseudo-opcodes to be generated, this test should be replaced with one that
; tests those opcodes.
;
; CHECK-LABEL: foo9:
; CHECK: ja
; CHECK-NOT: ja
define void @foo9(i32 %v1,
                  <8 x i1> %v12, <8 x i1> %v13,
                  <16 x i1> %v22, <16 x i1> %v23,
                  <32 x i1> %v32, <32 x i1> %v33,
                  <64 x i1> %v42, <64 x i1> %v43,
                  i8 * %dst) nounwind {
entry:
  %add.ptr11 = getelementptr inbounds i8, i8* %dst, i32 0
  %a11 = bitcast i8* %add.ptr11 to <8 x i1>*

  %add.ptr21 = getelementptr inbounds i8, i8* %dst, i32 4
  %a21 = bitcast i8* %add.ptr21 to <16 x i1>*

  %add.ptr31 = getelementptr inbounds i8, i8* %dst, i32 8
  %a31 = bitcast i8* %add.ptr31 to <32 x i1>*

  %add.ptr41 = getelementptr inbounds i8, i8* %dst, i32 16
  %a41 = bitcast i8* %add.ptr41 to <64 x i1>*

  ; These operations are necessary, because select of two single use loads
  ; ends up getting optimized into a select of two leas, followed by a
  ; single load of the selected address.
  %t13 = xor <8 x i1> %v13, %v12
  %t23 = xor <16 x i1> %v23, %v22
  %t33 = xor <32 x i1> %v33, %v32
  %t43 = xor <64 x i1> %v43, %v42

  %cmp = icmp ugt i32 %v1, 31
  %t11 = select i1 %cmp, <8 x i1> %v12, <8 x i1> %t13
  %t21 = select i1 %cmp, <16 x i1> %v22, <16 x i1> %t23
  %t31 = select i1 %cmp, <32 x i1> %v32, <32 x i1> %t33
  %t41 = select i1 %cmp, <64 x i1> %v42, <64 x i1> %t43

  store <8 x i1> %t11, <8 x i1>* %a11, align 16
  store <16 x i1> %t21, <16 x i1>* %a21, align 4
  store <32 x i1> %t31, <32 x i1>* %a31, align 8
  store <64 x i1> %t41, <64 x i1>* %a41, align 16

  ret void
}
