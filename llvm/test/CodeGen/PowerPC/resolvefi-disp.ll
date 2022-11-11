; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -print-after=localstackalloc <%s >%t 2>&1 && FileCheck <%t %s

; Due to a bug in isFrameOffsetLegal we ended up with resolveFrameIndex creating
; addresses with out-of-range displacements.  Verify that this no longer happens.
; CHECK-NOT: LD {{3276[8-9]}}
; CHECK-NOT: LD {{327[7-9][0-9]}}
; CHECK-NOT: LD {{32[8-9][0-9][0-9]}}
; CHECK-NOT: LD {{3[3-9][0-9][0-9][0-9]}}
; CHECK-NOT: LD {{[4-9][0-9][0-9][0-9][0-9]}}
; CHECK-NOT: LD {{[1-9][0-9][0-9][0-9][0-9][0-9]+}}

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%struct.S2760 = type { <2 x float>, %struct.anon, i32, [28 x i8] }
%struct.anon = type { [11 x %struct.anon.0], i64, [6 x { i64, i64 }], [24 x i8] }
%struct.anon.0 = type { [30 x %union.U4DI], i8, [0 x i16], [30 x i8] }
%union.U4DI = type { <4 x i64> }

@s2760 = external global %struct.S2760
@fails = external global i32

define void @check2760(ptr noalias sret(%struct.S2760) %agg.result, ptr byval(%struct.S2760) align 16, ptr %arg1, ptr byval(%struct.S2760) align 16) {
entry:
  %arg0 = alloca %struct.S2760, align 32
  %arg2 = alloca %struct.S2760, align 32
  %arg1.addr = alloca ptr, align 8
  %ret = alloca %struct.S2760, align 32
  %b1 = alloca %struct.S2760, align 32
  %b2 = alloca %struct.S2760, align 32
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %arg0, ptr align 16 %0, i64 11104, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %arg2, ptr align 16 %1, i64 11104, i1 false)
  store ptr %arg1, ptr %arg1.addr, align 8
  call void @llvm.memset.p0.i64(ptr align 32 %ret, i8 0, i64 11104, i1 false)
  call void @llvm.memset.p0.i64(ptr align 32 %b1, i8 0, i64 11104, i1 false)
  call void @llvm.memset.p0.i64(ptr align 32 %b2, i8 0, i64 11104, i1 false)
  %b = getelementptr inbounds %struct.S2760, ptr %arg0, i32 0, i32 1
  %g = getelementptr inbounds %struct.anon, ptr %b, i32 0, i32 1
  %2 = load i64, ptr %g, align 8
  %3 = load i64, ptr getelementptr inbounds (%struct.S2760, ptr @s2760, i32 0, i32 1, i32 1), align 8
  %cmp = icmp ne i64 %2, %3
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %4 = load i32, ptr @fails, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr @fails, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %5 = load i64, ptr getelementptr inbounds (%struct.S2760, ptr @s2760, i32 0, i32 1, i32 1), align 8
  %b3 = getelementptr inbounds %struct.S2760, ptr %ret, i32 0, i32 1
  %g4 = getelementptr inbounds %struct.anon, ptr %b3, i32 0, i32 1
  store i64 %5, ptr %g4, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 32 %agg.result, ptr align 32 %ret, i64 11104, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

