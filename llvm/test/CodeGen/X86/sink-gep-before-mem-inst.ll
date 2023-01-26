; RUN: opt -opaque-pointers=0 < %s -S -codegenprepare -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define i64 @test.after(i8 addrspace(1)* readonly align 8) {
; CHECK-LABEL: test.after
; CHECK: sunkaddr
entry:
  %.0 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 8
  %addr = bitcast i8 addrspace(1)* %.0 to i32 addrspace(1)*
  br label %header

header:
  %addr.in.loop = phi i32 addrspace(1)* [ %addr, %entry ], [ %addr.after, %header ]
  %local_2_ = phi i64 [ 0, %entry ], [ %.9, %header ]
  %.7 = load i32, i32 addrspace(1)* %addr.in.loop, align 8
  fence acquire
  %.1 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 8
  %addr.after = bitcast i8 addrspace(1)* %.1 to i32 addrspace(1)*
  %.8 = sext i32 %.7 to i64
  %.9 = add i64 %local_2_, %.8
  %not. = icmp sgt i64 %.9, 999
  br i1 %not., label %exit, label %header

exit:
  ret i64 %.9
}
