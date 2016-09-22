; Test basic EfficiencySanitizer struct field count instrumentation with -esan-small-binary
;
; RUN: opt < %s -esan -esan-cache-frag -esan-aux-field-info=false -S | FileCheck %s

%struct.A = type { i32, i32 }
%union.U = type { double }
%struct.C = type { %struct.anon, %union.anon, [10 x i8] }
%struct.anon = type { i32, i32 }
%union.anon = type { double }

; CHECK:        @0 = private unnamed_addr constant [8 x i8] c"<stdin>\00", align 1
; CHECK-NEXT:   @1 = private unnamed_addr constant [17 x i8] c"struct.A$2$11$11\00", align 1
; CHECK-NEXT:   @"struct.A$2$11$11" = weak global [3 x i64] zeroinitializer
; CHECK-NEXT:   @2 = private unnamed_addr constant [12 x i8] c"union.U$1$3\00", align 1
; CHECK-NEXT:   @"union.U$1$3" = weak global [2 x i64] zeroinitializer
; CHECK-NEXT:   @3 = private unnamed_addr constant [20 x i8] c"struct.C$3$14$13$13\00", align 1
; CHECK-NEXT:   @"struct.C$3$14$13$13" = weak global [4 x i64] zeroinitializer
; CHECK-NEXT:   @4 = private unnamed_addr constant [20 x i8] c"struct.anon$2$11$11\00", align 1
; CHECK-NEXT:   @"struct.anon$2$11$11" = weak global [3 x i64] zeroinitializer
; CHECK-NEXT:   @5 = private unnamed_addr constant [15 x i8] c"union.anon$1$3\00", align 1
; CHECK-NEXT:   @"union.anon$1$3" = weak global [2 x i64] zeroinitializer
; CHECK-NEXT:   @6 = internal global [5 x { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* }] [{ i8*, i32, i32, i32*, i32*, i8**, i64*, i64* } { i8* getelementptr inbounds ([17 x i8], [17 x i8]* @1, i32 0, i32 0), i32 8, i32 2, i32* null, i32* null, i8** null, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.A$2$11$11", i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.A$2$11$11", i32 0, i32 2) }, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* } { i8* getelementptr inbounds ([12 x i8], [12 x i8]* @2, i32 0, i32 0), i32 8, i32 1, i32* null, i32* null, i8** null, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"union.U$1$3", i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"union.U$1$3", i32 0, i32 1) }, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* } { i8* getelementptr inbounds ([20 x i8], [20 x i8]* @3, i32 0, i32 0), i32 32, i32 3, i32* null, i32* null, i8** null, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 0), i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3) }, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* } { i8* getelementptr inbounds ([20 x i8], [20 x i8]* @4, i32 0, i32 0), i32 8, i32 2, i32* null, i32* null, i8** null, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.anon$2$11$11", i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.anon$2$11$11", i32 0, i32 2) }, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* } { i8* getelementptr inbounds ([15 x i8], [15 x i8]* @5, i32 0, i32 0), i32 8, i32 1, i32* null, i32* null, i8** null, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"union.anon$1$3", i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* @"union.anon$1$3", i32 0, i32 1) }]
; CHECK-NEXT:   @7 = internal constant { i8*, i32, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* }* } { i8* getelementptr inbounds ([8 x i8], [8 x i8]* @0, i32 0, i32 0), i32 5, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* }* getelementptr inbounds ([5 x { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* }], [5 x { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* }]* @6, i32 0, i32 0) }

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
; CHECK: @llvm.global_dtors = {{.*}}@esan.module_dtor

; CHECK:        %a = alloca %struct.A, align 4
; CHECK-NEXT:   %u = alloca %union.U, align 8
; CHECK-NEXT:   %c = alloca [2 x %struct.C], align 16
; CHECK-NEXT:   %k = alloca %struct.A*, align 8
; CHECK-NEXT:   %0 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.A$2$11$11", i32 0, i32 0)
; CHECK-NEXT:   %1 = add i64 %0, 1
; CHECK-NEXT:   store i64 %1, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.A$2$11$11", i32 0, i32 0)
; CHECK-NEXT:   %x = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
; CHECK-NEXT:   %2 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.A$2$11$11", i32 0, i32 1)
; CHECK-NEXT:   %3 = add i64 %2, 1
; CHECK-NEXT:   store i64 %3, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.A$2$11$11", i32 0, i32 1)
; CHECK-NEXT:   %y = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 1
; CHECK-NEXT:   %f = bitcast %union.U* %u to float*
; CHECK-NEXT:   %d = bitcast %union.U* %u to double*
; CHECK-NEXT:   %4 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %5 = add i64 %4, 1
; CHECK-NEXT:   store i64 %5, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %arrayidx = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
; CHECK-NEXT:   %6 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 0)
; CHECK-NEXT:   %7 = add i64 %6, 1
; CHECK-NEXT:   store i64 %7, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 0)
; CHECK-NEXT:   %cs = getelementptr inbounds %struct.C, %struct.C* %arrayidx, i32 0, i32 0
; CHECK-NEXT:   %8 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.anon$2$11$11", i32 0, i32 0)
; CHECK-NEXT:   %9 = add i64 %8, 1
; CHECK-NEXT:   store i64 %9, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.anon$2$11$11", i32 0, i32 0)
; CHECK-NEXT:   %x1 = getelementptr inbounds %struct.anon, %struct.anon* %cs, i32 0, i32 0
; CHECK-NEXT:   %10 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %11 = add i64 %10, 1
; CHECK-NEXT:   store i64 %11, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %arrayidx2 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 1
; CHECK-NEXT:   %12 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 0)
; CHECK-NEXT:   %13 = add i64 %12, 1
; CHECK-NEXT:   store i64 %13, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 0)
; CHECK-NEXT:   %cs3 = getelementptr inbounds %struct.C, %struct.C* %arrayidx2, i32 0, i32 0
; CHECK-NEXT:   %14 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.anon$2$11$11", i32 0, i32 1)
; CHECK-NEXT:   %15 = add i64 %14, 1
; CHECK-NEXT:   store i64 %15, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @"struct.anon$2$11$11", i32 0, i32 1)
; CHECK-NEXT:   %y4 = getelementptr inbounds %struct.anon, %struct.anon* %cs3, i32 0, i32 1
; CHECK-NEXT:   %16 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %17 = add i64 %16, 1
; CHECK-NEXT:   store i64 %17, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %arrayidx5 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
; CHECK-NEXT:   %18 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 1)
; CHECK-NEXT:   %19 = add i64 %18, 1
; CHECK-NEXT:   store i64 %19, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 1)
; CHECK-NEXT:   %cu = getelementptr inbounds %struct.C, %struct.C* %arrayidx5, i32 0, i32 1
; CHECK-NEXT:   %f6 = bitcast %union.anon* %cu to float*
; CHECK-NEXT:   %20 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %21 = add i64 %20, 1
; CHECK-NEXT:   store i64 %21, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %arrayidx7 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 1
; CHECK-NEXT:   %22 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 1)
; CHECK-NEXT:   %23 = add i64 %22, 1
; CHECK-NEXT:   store i64 %23, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 1)
; CHECK-NEXT:   %cu8 = getelementptr inbounds %struct.C, %struct.C* %arrayidx7, i32 0, i32 1
; CHECK-NEXT:   %d9 = bitcast %union.anon* %cu8 to double*
; CHECK-NEXT:   %24 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %25 = add i64 %24, 1
; CHECK-NEXT:   store i64 %25, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 3)
; CHECK-NEXT:   %arrayidx10 = getelementptr inbounds [2 x %struct.C], [2 x %struct.C]* %c, i64 0, i64 0
; CHECK-NEXT:   %26 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 2)
; CHECK-NEXT:   %27 = add i64 %26, 1
; CHECK-NEXT:   store i64 %27, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @"struct.C$3$14$13$13", i32 0, i32 2)
; CHECK-NEXT:   %c11 = getelementptr inbounds %struct.C, %struct.C* %arrayidx10, i32 0, i32 2
; CHECK-NEXT:   %arrayidx12 = getelementptr inbounds [10 x i8], [10 x i8]* %c11, i64 0, i64 2
; CHECK-NEXT:   %k1 = load %struct.A*, %struct.A** %k, align 8
; CHECK-NEXT:   %arrayidx13 = getelementptr inbounds %struct.A, %struct.A* %k1, i64 0
; CHECK-NEXT:   ret i32 0

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Top-level:

; CHECK: define internal void @esan.module_ctor()
; CHECK: call void @__esan_init(i32 1, i8* bitcast ({ i8*, i32, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* }* }* @7 to i8*))
; CHECK: define internal void @esan.module_dtor()
; CHECK: call void @__esan_exit(i8* bitcast ({ i8*, i32, { i8*, i32, i32, i32*, i32*, i8**, i64*, i64* }* }* @7 to i8*))
