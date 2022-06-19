; RUN: opt < %s -passes=sroa -S | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.ld2 = type { [2 x ppc_fp128] }
declare void @bar(i8*, [2 x i128])

define void @foo(i8* %v) #0 {
entry:
  %v.addr = alloca i8*, align 8
  %z = alloca %struct.ld2, align 16
  store i8* %v, i8** %v.addr, align 8
  %dat = getelementptr inbounds %struct.ld2, %struct.ld2* %z, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x ppc_fp128], [2 x ppc_fp128]* %dat, i32 0, i64 0
  store ppc_fp128 0xM403B0000000000000000000000000000, ppc_fp128* %arrayidx, align 16
  %dat1 = getelementptr inbounds %struct.ld2, %struct.ld2* %z, i32 0, i32 0
  %arrayidx2 = getelementptr inbounds [2 x ppc_fp128], [2 x ppc_fp128]* %dat1, i32 0, i64 1
  store ppc_fp128 0xM4093B400000000000000000000000000, ppc_fp128* %arrayidx2, align 16
  %0 = load i8*, i8** %v.addr, align 8
  %coerce.dive = getelementptr %struct.ld2, %struct.ld2* %z, i32 0, i32 0
  %1 = bitcast [2 x ppc_fp128]* %coerce.dive to [2 x i128]*
  %2 = load [2 x i128], [2 x i128]* %1, align 1
  call void @bar(i8* %0, [2 x i128] %2)
  ret void
}

; CHECK-LABEL: @foo
; CHECK-NOT: i128 4628293042053316608
; CHECK-NOT: i128 4653260752096854016
; CHECK-DAG: bitcast ppc_fp128 0xM403B0000000000000000000000000000 to i128
; CHECK-DAG: bitcast ppc_fp128 0xM4093B400000000000000000000000000 to i128
; CHECK: call void @bar(i8* %v, [2 x i128]
; CHECK: ret void

attributes #0 = { nounwind }

