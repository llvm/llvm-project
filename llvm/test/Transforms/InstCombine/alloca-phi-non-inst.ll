; RUN: opt < %s -O1 -S | FileCheck %s

@__const.var1 = addrspace(4) constant <{ ptr addrspace(5), ptr addrspace(5), ptr addrspace(5), ptr addrspace(5) }> <{ ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)) }>
define void @test(i32 %0, i1 %tobool, ptr addrspace(4) %const_src) {
; CHECK:  entry:
; CHECK-NEXT:    br label %BS_LABEL_3
; CHECK:       BS_LABEL_3:
; CHECK-NEXT:    br label %BS_LABEL_3
entry:
  %l_632 = alloca [4 x ptr addrspace(5)], align 4, addrspace(5)
  switch i32 %0, label %sw.epilog [
    i32 1, label %BS_LABEL_3
    i32 0, label %BS_LABEL_3
  ]

sw.epilog:                                        ; preds = %entry
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) %l_632, ptr addrspace(4) @__const.var1, i64 16, i1 false)
  %arrayidx = getelementptr inbounds [4 x ptr addrspace(5)], ptr addrspace(5) %l_632, i64 0, i64 3
  br i1 %tobool, label %BS_LABEL_7, label %BS_LABEL_3

BS_LABEL_7:                                       ; preds = %BS_LABEL_3, %sw.epilog
  %l_631.1 = phi ptr addrspace(5) [ %arrayidx, %sw.epilog ], [ %l_631.0, %BS_LABEL_3 ]
  br label %BS_LABEL_3

BS_LABEL_3:                                       ; preds = %BS_LABEL_7, %sw.epilog, %entry, %entry
  %l_631.0 = phi ptr addrspace(5) [ %l_631.1, %BS_LABEL_7 ], [ %arrayidx, %sw.epilog ], [ undef, %entry ], [ undef, %entry ]
  %cmp = icmp ugt ptr addrspace(5) %l_631.0, null
  br label %BS_LABEL_7
}