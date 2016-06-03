; Test basic EfficiencySanitizer struct field count instrumentation.
;
; RUN: opt < %s -esan -esan-cache-frag -esan-instrument-loads-and-stores=false -esan-instrument-memintrinsics=false -S | FileCheck %s

%struct.A = type { i32, i32 }
%union.U = type { double }
%struct.C = type { %struct.anon, %union.anon, [10 x i8] }
%struct.anon = type { i32, i32 }
%union.anon = type { double }

define i32 @main() {
entry:
  %a = alloca %struct.A, align 4
  %u = alloca %union.U, align 8
  %c = alloca [2 x %struct.C], align 16
  %k = alloca %struct.A*, align 8
  %x = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
  %y = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 1
  %f = bitcast %union.U* %u to float*
  %d = bitcast %union.U* %u to double*
  %arrayidx = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
  %cs = getelementptr inbounds %struct.C, %struct.C* %arrayidx, i32 0, i32 0
  %x1 = getelementptr inbounds %struct.anon, %struct.anon* %cs, i32 0, i32 0
  %arrayidx2 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 1
  %cs3 = getelementptr inbounds %struct.C, %struct.C* %arrayidx2, i32 0, i32 0
  %y4 = getelementptr inbounds %struct.anon, %struct.anon* %cs3, i32 0, i32 1
  %arrayidx5 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
  %cu = getelementptr inbounds %struct.C, %struct.C* %arrayidx5, i32 0, i32 1
  %f6 = bitcast %union.anon* %cu to float*
  %arrayidx7 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 1
  %cu8 = getelementptr inbounds %struct.C, %struct.C* %arrayidx7, i32 0, i32 1
  %d9 = bitcast %union.anon* %cu8 to double*
  %arrayidx10 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
  %c11 = getelementptr inbounds %struct.C, %struct.C* %arrayidx10, i32 0, i32 2
  %arrayidx12 = getelementptr inbounds [10 x i8], [10 x i8]* %c11, i64 0, i64 2
  %k1 = load %struct.A*, %struct.A** %k, align 8
  %arrayidx13 = getelementptr inbounds %struct.A, %struct.A* %k1, i64 0
  ret i32 0
}

; CHECK: @llvm.global_ctors = {{.*}}@esan.module_ctor

; CHECK:        %a = alloca %struct.A, align 4
; CHECK-NEXT:   %u = alloca %union.U, align 8
; CHECK-NEXT:   %c = alloca [2 x %struct.C], align 16
; CHECK-NEXT:   %k = alloca %struct.A*, align 8
; CHECK-NEXT:   %0 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.A#2#11#11", i32 0, i32 0)
; CHECK-NEXT:   %1 = add i64 %0, 1
; CHECK-NEXT:   store i64 %1, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.A#2#11#11", i32 0, i32 0)
; CHECK-NEXT:   %x = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
; CHECK-NEXT:   %2 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.A#2#11#11", i32 0, i32 1)
; CHECK-NEXT:   %3 = add i64 %2, 1
; CHECK-NEXT:   store i64 %3, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.A#2#11#11", i32 0, i32 1)
; CHECK-NEXT:   %y = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 1
; CHECK-NEXT:   %f = bitcast %union.U* %u to float*
; CHECK-NEXT:   %d = bitcast %union.U* %u to double*
; CHECK-NEXT:   %arrayidx = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
; CHECK-NEXT:   %4 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 0)
; CHECK-NEXT:   %5 = add i64 %4, 1
; CHECK-NEXT:   store i64 %5, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 0)
; CHECK-NEXT:   %cs = getelementptr inbounds %struct.C, %struct.C* %arrayidx, i32 0, i32 0
; CHECK-NEXT:   %6 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.anon#2#11#11", i32 0, i32 0)
; CHECK-NEXT:   %7 = add i64 %6, 1
; CHECK-NEXT:   store i64 %7, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.anon#2#11#11", i32 0, i32 0)
; CHECK-NEXT:   %x1 = getelementptr inbounds %struct.anon, %struct.anon* %cs, i32 0, i32 0
; CHECK-NEXT:   %arrayidx2 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 1
; CHECK-NEXT:   %8 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 0)
; CHECK-NEXT:   %9 = add i64 %8, 1
; CHECK-NEXT:   store i64 %9, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 0)
; CHECK-NEXT:   %cs3 = getelementptr inbounds %struct.C, %struct.C* %arrayidx2, i32 0, i32 0
; CHECK-NEXT:   %10 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.anon#2#11#11", i32 0, i32 1)
; CHECK-NEXT:   %11 = add i64 %10, 1
; CHECK-NEXT:   store i64 %11, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"struct.anon#2#11#11", i32 0, i32 1)
; CHECK-NEXT:   %y4 = getelementptr inbounds %struct.anon, %struct.anon* %cs3, i32 0, i32 1
; CHECK-NEXT:   %arrayidx5 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
; CHECK-NEXT:   %12 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 1)
; CHECK-NEXT:   %13 = add i64 %12, 1
; CHECK-NEXT:   store i64 %13, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 1)
; CHECK-NEXT:   %cu = getelementptr inbounds %struct.C, %struct.C* %arrayidx5, i32 0, i32 1
; CHECK-NEXT:   %f6 = bitcast %union.anon* %cu to float*
; CHECK-NEXT:   %arrayidx7 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 1
; CHECK-NEXT:   %14 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 1)
; CHECK-NEXT:   %15 = add i64 %14, 1
; CHECK-NEXT:   store i64 %15, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 1)
; CHECK-NEXT:   %cu8 = getelementptr inbounds %struct.C, %struct.C* %arrayidx7, i32 0, i32 1
; CHECK-NEXT:   %d9 = bitcast %union.anon* %cu8 to double*
; CHECK-NEXT:   %arrayidx10 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
; CHECK-NEXT:   %16 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 2)
; CHECK-NEXT:   %17 = add i64 %16, 1
; CHECK-NEXT:   store i64 %17, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.C#3#14#13#13", i32 0, i32 2)
; CHECK-NEXT:   %c11 = getelementptr inbounds %struct.C, %struct.C* %arrayidx10, i32 0, i32 2
; CHECK-NEXT:   %arrayidx12 = getelementptr inbounds [10 x i8], [10 x i8]* %c11, i64 0, i64 2
; CHECK-NEXT:   %k1 = load %struct.A*, %struct.A** %k, align 8
; CHECK-NEXT:   %arrayidx13 = getelementptr inbounds %struct.A, %struct.A* %k1, i64 0
; CHECK-NEXT:   ret i32 0
