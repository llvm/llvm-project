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

define void @atomicLoadsBothWriteAndReadMem(%MyStruct *%a, %MyStruct *%b, %MyStruct *%lim) {
entry:
  br label %loop

loop:
  %0 = phi %MyStruct* [ %a, %entry ], [ %ainc, %loop ]
  %1 = phi %MyStruct* [ %b, %entry ], [ %binc, %loop ]
  %2 = getelementptr %MyStruct, %MyStruct* %1, i64 0, i32 0
  %3 = load i32, i32* %2, align 8
  %4 = getelementptr inbounds %MyStruct, %MyStruct* %0, i64 0, i32 0
  store i32 %3, i32* %4, align 8
  %5 = getelementptr inbounds %MyStruct, %MyStruct* %1, i64 0, i32 1, i32 0, i32 0
  %6 = load atomic i32, i32* %5 monotonic, align 4
  %7 = getelementptr inbounds %MyStruct, %MyStruct* %0, i64 0, i32 1, i32 0, i32 0
  store atomic i32 %6, i32* %7 monotonic, align 4
  %8 = getelementptr inbounds %MyStruct, %MyStruct* %1, i64 0, i32 2, i32 0
  %9 = getelementptr inbounds %MyStruct, %MyStruct* %0, i64 0, i32 2, i32 0
  %10 = load i64, i64* %8, align 8
  store i64 %10, i64* %9, align 8
  %binc = getelementptr inbounds %MyStruct, %MyStruct* %1, i64 1
  %ainc = getelementptr inbounds %MyStruct, %MyStruct* %0, i64 1
  %cond = icmp eq %MyStruct* %binc, %lim
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}
