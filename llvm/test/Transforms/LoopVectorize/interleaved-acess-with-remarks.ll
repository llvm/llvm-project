; RUN: opt -S -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true -runtime-memory-check-threshold=24 --pass-remarks=loop-vectorize < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"


; This only tests that asking for remarks doesn't lead to compiler crashing
; (or timing out). We just check for output. To be sure, we also check we didn't
; vectorize.
; CHECK-LABEL: @atomicLoadsBothWriteAndReadMem
; CHECK-NOT: <{{[0-9]+}} x i8>

%"struct.std::__atomic_base" = type { i32 }
%"struct.std::atomic" = type { %"struct.std::__atomic_base" }
%union.anon = type { i64 }
%MyStruct = type { i32, %"struct.std::atomic", %union.anon }

define void @atomicLoadsBothWriteAndReadMem(ptr %a, ptr %b, ptr %lim) {
entry:
  br label %loop

loop:
  %0 = phi ptr [ %a, %entry ], [ %ainc, %loop ]
  %1 = phi ptr [ %b, %entry ], [ %binc, %loop ]
  %2 = load i32, ptr %1, align 8
  store i32 %2, ptr %0, align 8
  %3 = getelementptr inbounds %MyStruct, ptr %1, i64 0, i32 1, i32 0, i32 0
  %4 = load atomic i32, ptr %3 monotonic, align 4
  %5 = getelementptr inbounds %MyStruct, ptr %0, i64 0, i32 1, i32 0, i32 0
  store atomic i32 %4, ptr %5 monotonic, align 4
  %6 = getelementptr inbounds %MyStruct, ptr %1, i64 0, i32 2, i32 0
  %7 = getelementptr inbounds %MyStruct, ptr %0, i64 0, i32 2, i32 0
  %8 = load i64, ptr %6, align 8
  store i64 %8, ptr %7, align 8
  %binc = getelementptr inbounds %MyStruct, ptr %1, i64 1
  %ainc = getelementptr inbounds %MyStruct, ptr %0, i64 1
  %cond = icmp eq ptr %binc, %lim
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}
