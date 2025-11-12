; Note: Make sure that instrumention intrinsic is after entry alloca.
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -sampled-instrumentation -S | FileCheck %s --check-prefixes=SAMPLE

%struct.A = type { i32, [0 x i32] }
%struct.B = type { i32, [0 x double] }

; CHECK-LABEL: @foo()
; CHECK-NEXT:   %1 = alloca %struct.A
; CHECK-NEXT:   %2 = alloca %struct.B
; CHECK-NEXT:   call void @llvm.instrprof.increment(ptr @__profn_foo

; SAMPLE: @foo()
; SAMPLE-NEXT:  %1 = alloca %struct.A
; SAMPLE-NEXT:  %2 = alloca %struct.B
; SAMPLE-NEXT:  %[[v:[0-9]+]] = load i16, ptr @__llvm_profile_sampling
; SAMPLE-NEXT:  {{.*}} = icmp ule i16 %[[v]], 199

define dso_local double @foo() {
  %1 = alloca %struct.A, align 4
  %2 = alloca %struct.B, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  call void @bar(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %3 = load i32, ptr %1, align 4
  %4 = icmp sgt i32 %3, 0
  br i1 %4, label %5, label %21

5:
  %6 = getelementptr inbounds i8, ptr %1, i64 4
  %7 = getelementptr inbounds i8, ptr %2, i64 8
  %8 = zext nneg i32 %3 to i64
  br label %9

9:
  %10 = phi i64 [ 0, %5 ], [ %19, %9 ]
  %11 = phi double [ 0.000000e+00, %5 ], [ %18, %9 ]
  %12 = getelementptr inbounds [0 x i32], ptr %6, i64 0, i64 %10
  %13 = load i32, ptr %12, align 4
  %14 = sitofp i32 %13 to double
  %15 = getelementptr inbounds [0 x double], ptr %7, i64 0, i64 %10
  %16 = load double, ptr %15, align 8
  %17 = fadd double %16, %14
  %18 = fadd double %11, %17
  %19 = add nuw nsw i64 %10, 1
  %20 = icmp eq i64 %19, %8
  br i1 %20, label %21, label %9

21:
  %22 = phi double [ 0.000000e+00, %0 ], [ %18, %9 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  ret double %22
}

declare void @bar(ptr noundef, ptr noundef)
