; RUN: opt -gvn -S < %s | FileCheck %s

; Make sure we don't crash when analyzing an addrspacecast in
; GetPointerBaseWithConstantOffset()

target datalayout = "e-p:32:32-p4:64:64"

define i32 @addrspacecast-crash() {
; CHECK-LABEL: @addrspacecast-crash
; CHECK: %tmp = alloca [25 x i64]
; CHECK: %tmp1 = getelementptr inbounds [25 x i64], [25 x i64]* %tmp, i32 0, i32 0
; CHECK: %tmp2 = addrspacecast i64* %tmp1 to <8 x i64> addrspace(4)*
; CHECK: store <8 x i64> zeroinitializer, <8 x i64> addrspace(4)* %tmp2
; CHECK-NOT: load
bb:
  %tmp = alloca [25 x i64]
  %tmp1 = getelementptr inbounds [25 x i64], [25 x i64]* %tmp, i32 0, i32 0
  %tmp2 = addrspacecast i64* %tmp1 to <8 x i64> addrspace(4)*
  %tmp3 = getelementptr inbounds <8 x i64>, <8 x i64> addrspace(4)* %tmp2, i64 0
  store <8 x i64> zeroinitializer, <8 x i64> addrspace(4)* %tmp3
  %tmp4 = getelementptr inbounds [25 x i64], [25 x i64]* %tmp, i32 0, i32 0
  %tmp5 = addrspacecast i64* %tmp4 to i32 addrspace(4)*
  %tmp6 = getelementptr inbounds i32, i32 addrspace(4)* %tmp5, i64 10
  %tmp7 = load i32, i32 addrspace(4)* %tmp6, align 4
  ret i32 %tmp7
}
