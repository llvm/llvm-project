; RUN: opt -S -passes='require<profile-summary>,function(codegenprepare)' -disable-complex-addr-modes=false -disable-cgp-delete-phis %s | FileCheck %s --check-prefix=CHECK
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
entry:
  %0 = getelementptr inbounds i64, ptr null, i64 undef
  br label %start

start:
  %val1 = phi ptr [ %0, %entry ], [ %val4, %exit ]
  %val2 = phi ptr [ null, %entry ], [ %val5, %exit ]
  br i1 false, label %slowpath, label %exit

slowpath:
  %elem1 = getelementptr inbounds i64, ptr undef, i64 undef
  br label %exit

exit:
; CHECK: sunkaddr
  %val3 = phi ptr [ undef, %slowpath ], [ %val2, %start ]
  %val4 = phi ptr [ %elem1, %slowpath ], [ %val1, %start ]
  %val5 = phi ptr [ undef, %slowpath ], [ %val2, %start ]
  %loadx = load i64, ptr %val4, align 8
  br label %start
}
