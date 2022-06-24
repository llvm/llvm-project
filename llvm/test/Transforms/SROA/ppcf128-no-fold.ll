; RUN: opt < %s -passes=sroa -S | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.ld2 = type { [2 x ppc_fp128] }
declare void @bar(ptr, [2 x i128])

define void @foo(ptr %v) #0 {
entry:
  %v.addr = alloca ptr, align 8
  %z = alloca %struct.ld2, align 16
  store ptr %v, ptr %v.addr, align 8
  store ppc_fp128 0xM403B0000000000000000000000000000, ptr %z, align 16
  %arrayidx2 = getelementptr inbounds [2 x ppc_fp128], ptr %z, i32 0, i64 1
  store ppc_fp128 0xM4093B400000000000000000000000000, ptr %arrayidx2, align 16
  %0 = load ptr, ptr %v.addr, align 8
  %1 = load [2 x i128], ptr %z, align 1
  call void @bar(ptr %0, [2 x i128] %1)
  ret void
}

; CHECK-LABEL: @foo
; CHECK-NOT: i128 4628293042053316608
; CHECK-NOT: i128 4653260752096854016
; CHECK-DAG: bitcast ppc_fp128 0xM403B0000000000000000000000000000 to i128
; CHECK-DAG: bitcast ppc_fp128 0xM4093B400000000000000000000000000 to i128
; CHECK: call void @bar(ptr %v, [2 x i128]
; CHECK: ret void

attributes #0 = { nounwind }

