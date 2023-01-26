; Test that loops with sufficient registers do not reload or spill on
; stack. These cases include calls and it is necessary to have the GR128 /
; FP128 registers part of the callee saved registers list in order to avoid
; spilling / reloading.
;
; RUN: llc -switch-peel-threshold=101 < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

%0 = type { ptr, ptr, ptr, i32, ptr, i64, i64, i64, i64, i64, i64, %2, %5, %7 }
%1 = type { i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%2 = type { i64, i64, ptr }
%3 = type { ptr, i64 }
%4 = type { i64, ptr }
%5 = type { i64, i64, ptr }
%6 = type { i64, ptr, i32, i64, ptr }
%7 = type { i64, i64, ptr }
%8 = type { i64, ptr, ptr, ptr, i64, ptr, %5, i32, i64, i64 }

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

define void @fun0(ptr) {
; CHECK-LABEL: .LBB0_4
; CHECK: =>  This Inner Loop Header
; CHECK-NOT: 16-byte Folded Spill
; CHECK-NOT: 16-byte Folded Reload

  %2 = load i64, ptr undef, align 8
  %3 = udiv i64 128, %2
  %4 = mul i64 %3, %2
  %5 = load i64, ptr undef, align 8
  switch i32 undef, label %36 [
    i32 1, label %6
    i32 2, label %7
    i32 3, label %8
    i32 4, label %9
    i32 5, label %10
    i32 6, label %11
  ]

; <label>:6:                                      ; preds = %1
  br label %12

; <label>:7:                                      ; preds = %1
  br label %12

; <label>:8:                                      ; preds = %1
  unreachable

; <label>:9:                                      ; preds = %1
  unreachable

; <label>:10:                                     ; preds = %1
  unreachable

; <label>:11:                                     ; preds = %1
  unreachable

; <label>:12:                                     ; preds = %7, %6
  %13 = getelementptr inbounds %0, ptr %0, i64 0, i32 5
  br label %14

; <label>:14:                                     ; preds = %31, %12
  %15 = phi i64 [ undef, %31 ], [ %5, %12 ]
  %16 = phi i64 [ %35, %31 ], [ undef, %12 ]
  %17 = load i64, ptr %13, align 8
  %18 = icmp ult i64 %15, %17
  %19 = select i1 %18, i64 %15, i64 %17
  %20 = udiv i64 %19, %4
  %21 = icmp ugt i64 %20, 1
  %22 = select i1 %21, i64 %20, i64 1
  %23 = sub i64 %22, 0
  br label %24

; <label>:24:                                     ; preds = %24, %14
  %25 = phi i64 [ %23, %14 ], [ %27, %24 ]
  call void @llvm.memcpy.p0.p0.i64(ptr undef, ptr nonnull undef, i64 %4, i1 false)
  %26 = getelementptr inbounds i8, ptr null, i64 %4
  store ptr %26, ptr undef, align 8
  %27 = add i64 %25, -4
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %31, label %24

; <label>:29:                                     ; preds = %24
  br i1 undef, label %31, label %30

; <label>:30:                                     ; preds = %29
  call void @llvm.memcpy.p0.p0.i64(ptr %26, ptr nonnull undef, i64 %4, i1 false)
  br label %31

; <label>:31:                                     ; preds = %30, %29
  %32 = call signext i32 undef(ptr undef, i64 %16, i32 signext 8)
  %33 = icmp eq i64 undef, 0
  %34 = select i1 %33, i64 0, i64 %19
  %35 = add i64 %34, %16
  br i1 %33, label %36, label %14

; <label>:36:                                     ; preds = %31, %1
  ret void
}

declare fp128 @llvm.pow.f128(fp128, fp128)

define void @fun1(ptr) {
; CHECK-LABEL: .LBB1_2
; CHECK: =>This Inner Loop Header: Depth=1
; CHECK-NOT: 16-byte Folded Spill
; CHECK-NOT: 16-byte Folded Reload
; CHECK-LABEL: .LBB1_3

  br i1 undef, label %7, label %2

; <label>:2:                                      ; preds = %2, %1
  %3 = phi fp128 [ %5, %2 ], [ 0xL00000000000000000000000000000000, %1 ]
  %4 = tail call fp128 @llvm.pow.f128(fp128 0xL00000000000000000000000000000000, fp128 0xL00000000000000000000000000000000) #2
  %5 = fadd fp128 %3, %4
  %6 = icmp eq i64 undef, 0
  br i1 %6, label %7, label %2

; <label>:7:                                      ; preds = %2, %1
  %8 = phi fp128 [ 0xL00000000000000000000000000000000, %1 ], [ %5, %2 ]
  %9 = fadd fp128 0xL00000000000000000000000000000000, %8
  %10 = fadd fp128 0xL00000000000000000000000000000000, %9
  %11 = fadd fp128 0xL00000000000000000000000000000000, %10
  %12 = tail call fp128 @llvm.pow.f128(fp128 %11, fp128 0xL00000000000000000000000000000000) #2
  store fp128 %12, ptr %0, align 8
  ret void
}
