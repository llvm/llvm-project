; RUN: opt < %s -licm -S | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p4:64:64"

; Make sure isDereferenceableAndAlignePointer() doesn't crash when looking
; walking pointer defs with an addrspacecast that changes pointer size.
; CHECK-LABEL: @addrspacecast_crash
define void @addrspacecast_crash() {
bb:
  %tmp = alloca [256 x i32]
  br label %bb1

bb1:
  %tmp2 = getelementptr inbounds [256 x i32], [256 x i32]* %tmp, i32 0, i32 36
  %tmp3 = bitcast i32* %tmp2 to <4 x i32>*
  %tmp4 = addrspacecast <4 x i32>* %tmp3 to <4 x i32> addrspace(4)*
  %tmp5 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp4
  %tmp6 = xor <4 x i32> %tmp5, undef
  store <4 x i32> %tmp6, <4 x i32> addrspace(1)* undef
  br label %bb1
}
