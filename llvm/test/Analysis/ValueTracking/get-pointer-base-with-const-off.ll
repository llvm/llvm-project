; RUN: opt -passes=gvn -S < %s | FileCheck %s

; Make sure we don't crash when analyzing an addrspacecast in
; GetPointerBaseWithConstantOffset()

target datalayout = "e-p:32:32-p4:64:64"

define i32 @addrspacecast-crash() {
; CHECK-LABEL: @addrspacecast-crash
; CHECK: %tmp = alloca [25 x i64]
; CHECK: %tmp2 = addrspacecast ptr %tmp to ptr addrspace(4)
; CHECK: store <8 x i64> zeroinitializer, ptr addrspace(4) %tmp2
; CHECK-NOT: load
bb:
  %tmp = alloca [25 x i64]
  %tmp2 = addrspacecast ptr %tmp to ptr addrspace(4)
  store <8 x i64> zeroinitializer, ptr addrspace(4) %tmp2
  %tmp5 = addrspacecast ptr %tmp to ptr addrspace(4)
  %tmp6 = getelementptr inbounds i32, ptr addrspace(4) %tmp5, i64 10
  %tmp7 = load i32, ptr addrspace(4) %tmp6, align 4
  ret i32 %tmp7
}
