; RUN: opt < %s -passes=licm -S | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p4:64:64"

; Make sure isDereferenceableAndAlignePointer() doesn't crash when looking
; walking pointer defs with an addrspacecast that changes pointer size.
; CHECK-LABEL: @addrspacecast_crash
define void @addrspacecast_crash() {
bb:
  %tmp = alloca [256 x i32]
  br label %bb1

bb1:
  %tmp2 = getelementptr inbounds [256 x i32], ptr %tmp, i32 0, i32 36
  %tmp4 = addrspacecast ptr %tmp2 to ptr addrspace(4)
  %tmp5 = load <4 x i32>, ptr addrspace(4) %tmp4
  %tmp6 = xor <4 x i32> %tmp5, undef
  store <4 x i32> %tmp6, ptr addrspace(1) undef
  br label %bb1
}
