; SROA fails to rewrite allocs but does rewrite some phis and delete
; dead instructions. Ensure that this invalidates analyses required
; for other passes.
; RUN: opt < %s -passes=bdce,sroa,bdce -o %t -debug-pass-manager 2>&1 | FileCheck %s
; CHECK: Running pass: BDCEPass on H
; CHECK: Running analysis: DemandedBitsAnalysis on H
; CHECK: Running pass: SROAPass on H
; CHECK: Invalidating analysis: DemandedBitsAnalysis on H
; CHECK: Running pass: BDCEPass on H
; CHECK: Running analysis: DemandedBitsAnalysis on H

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-grtev4-linux-gnu"

%class.b = type { i64 }

declare void @D(%class.b* sret(%class.b), %class.b* dereferenceable(32)) local_unnamed_addr

; Function Attrs: nounwind
define hidden fastcc void @H(%class.b* noalias nocapture readnone, [2 x i64]) unnamed_addr {
  %3 = alloca %class.b, align 8
  %.sroa.0 = alloca i64, align 8
  store i64 0, i64* %.sroa.0, align 8
  %4 = extractvalue [2 x i64] %1, 1
  switch i64 %4, label %6 [
    i64 4, label %foo
    i64 5, label %5
  ]

; <label>:5:
  %.sroa.0.0..sroa_cast3 = bitcast i64* %.sroa.0 to i8**
  br label %12

; <label>:6:
  %7 = icmp ugt i64 %4, 5
  %.sroa.0.0..sroa_cast5 = bitcast i64* %.sroa.0 to i8**
  br i1 %7, label %8, label %12

; <label>:8:
  %9 = load i8, i8* inttoptr (i64 4 to i8*), align 4
  %10 = icmp eq i8 %9, 47
  %11 = select i1 %10, i64 5, i64 4
  br label %12

; <label>:12:
  %13 = phi i8** [ %.sroa.0.0..sroa_cast3, %5 ], [ %.sroa.0.0..sroa_cast5, %8 ], [ %.sroa.0.0..sroa_cast5, %6 ]
  %14 = phi i64 [ 4, %5 ], [ %11, %8 ], [ 4, %6 ]
  %15 = icmp ne i64 %4, 0
  %16 = icmp ugt i64 %4, %14
  %17 = and i1 %15, %16
  br i1 %17, label %18, label %a.exit

; <label>:18:
  %19 = tail call i8* @memchr(i8* undef, i32 signext undef, i64 undef)
  %20 = icmp eq i8* %19, null
  %21 = sext i1 %20 to i64
  br label %a.exit

a.exit:
  %22 = phi i64 [ -1, %12 ], [ %21, %18 ]
  %23 = load i8*, i8** %13, align 8
  %24 = sub nsw i64 %22, %14
  %25 = bitcast %class.b* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %25)
  %26 = icmp ult i64 %24, 2
  br i1 %26, label %G.exit, label %27

; <label>:27:
  %28 = getelementptr inbounds i8, i8* %23, i64 undef
  %29 = icmp eq i8* %28, null
  br i1 %29, label %30, label %31

; <label>:30:
  unreachable

; <label>:31:
  call void @D(%class.b* nonnull sret(%class.b) %3, %class.b* nonnull dereferenceable(32) undef)
  br label %G.exit

G.exit:
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %25)
  br label %foo

foo:
  ret void
}

; Function Attrs: nounwind readonly
declare i8* @memchr(i8*, i32 signext, i64) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
