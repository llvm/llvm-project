; Test that we don't crash.
; RUN: opt < %s -passes=alignment-from-assumptions -S

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7:8"

%"core::str::CharIndices.29.66.90.114.138.149.165.173.181.197.205.213.229.387.398" = type { [0 x i64], i64, [0 x i64], { ptr, ptr }, [0 x i64] }
%"unwind::libunwind::_Unwind_Exception.9.51.75.99.123.147.163.171.179.195.203.211.227.385.396" = type { [0 x i64], i64, [0 x i64], ptr, [0 x i64], [6 x i64], [0 x i64] }
%"unwind::libunwind::_Unwind_Context.10.52.76.100.124.148.164.172.180.196.204.212.228.386.397" = type { [0 x i8] }

define void @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h7b1d039c7ff5e1feE"(i1 %arg) {
start:
  %_15.i.i = alloca %"core::str::CharIndices.29.66.90.114.138.149.165.173.181.197.205.213.229.387.398", align 8, addrspace(5)
  br label %bb12.i.i

bb12.i.i:                                         ; preds = %start
  %0 = addrspacecast ptr addrspace(5) %_15.i.i to ptr
  %ptrint53.i.i = ptrtoint ptr %0 to i64
  %maskedptr54.i.i = and i64 %ptrint53.i.i, 7
  %maskcond55.i.i = icmp eq i64 %maskedptr54.i.i, 0
  call void @llvm.assume(i1 %maskcond55.i.i)
  br i1 %arg, label %bb20.i.i, label %bb3.i.i.i.i.i.preheader.i.i

bb3.i.i.i.i.i.preheader.i.i:                      ; preds = %bb12.i.i
  store i64 0, ptr %0, align 8
  unreachable

bb20.i.i:                                         ; preds = %bb12.i.i
  ret void
}

declare void @llvm.assume(i1)
