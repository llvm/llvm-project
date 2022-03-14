; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; This code causes the SCoP to be rejected because of an ERRORBLOCK
; assumption and made Polly crash (llvm.org/PR38219).
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define dso_local void @pr38219() {
start:
  %tmp1.i.i.i = icmp ne i64** null, null
  call void @llvm.assume(i1 %tmp1.i.i.i)
  %tmp1 = extractvalue { [0 x i64*]*, i64 } undef, 0
  %tmp.i1 = getelementptr inbounds [0 x i64*], [0 x i64*]* %tmp1, i64 0, i64 0
  br label %bb10.i

bb10.i:
  %_10.12.i = phi i64** [ %tmp.i1, %start ], [ undef, %_ZN4core3ptr13drop_in_place17hd1d510ec1955c343E.exit.i ]
  %tmp1.i.i2.i.i.i.i = load i64*, i64** %_10.12.i, align 8
  store i64 undef, i64* %tmp1.i.i2.i.i.i.i, align 1
  br label %bb3.i.i.i

bb3.i.i.i:
  store i64 0, i64* inttoptr (i64 8 to i64*), align 8
  br label %_ZN4core3ptr13drop_in_place17hd1d510ec1955c343E.exit.i

_ZN4core3ptr13drop_in_place17hd1d510ec1955c343E.exit.i:
  br i1 false, label %_ZN4core3ptr13drop_in_place17h76d4fbbcbbbe0ba5E.exit, label %bb10.i

_ZN4core3ptr13drop_in_place17h76d4fbbcbbbe0ba5E.exit:
  ret void
}

declare void @llvm.assume(i1)


; CHECK: Invalid Scop!
