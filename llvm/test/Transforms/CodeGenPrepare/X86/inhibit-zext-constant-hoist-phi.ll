; Make sure that if a phi with identical inputs gets created it gets undone by CodeGenPrepare.

; RUN: opt -codegenprepare -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__typeid__ZTS1S_global_addr = external hidden global [0 x i8], code_model "small"
@__typeid__ZTS1S_align = external hidden global [0 x i8], !absolute_symbol !0
@__typeid__ZTS1S_size_m1 = external hidden global [0 x i8], !absolute_symbol !1

; Check that we recover the third pair of zexts from the phi.

; CHECK: define void @f4
define void @f4(i1 noundef zeroext %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #1 {
  br i1 %0, label %5, label %18

5:
  %6 = load ptr, ptr %1, align 8
  %7 = ptrtoint ptr %6 to i64
  %8 = sub i64 %7, ptrtoint (ptr @__typeid__ZTS1S_global_addr to i64)
  ; CHECK: zext {{.*}} @__typeid__ZTS1S_align
  %9 = zext nneg i8 ptrtoint (ptr @__typeid__ZTS1S_align to i8) to i64
  %10 = lshr i64 %8, %9
  ; CHECK: zext {{.*}} @__typeid__ZTS1S_align
  %11 = zext nneg i8 sub (i8 64, i8 ptrtoint (ptr @__typeid__ZTS1S_align to i8)) to i64
  %12 = shl i64 %8, %11
  %13 = or i64 %10, %12
  %14 = icmp ugt i64 %13, ptrtoint (ptr @__typeid__ZTS1S_size_m1 to i64)
  br i1 %14, label %15, label %16

15:
  tail call void @llvm.ubsantrap(i8 2) #5
  unreachable

16:
  %17 = load ptr, ptr %6, align 8
  tail call void %17(ptr noundef nonnull align 8 dereferenceable(8) %1)
  br label %31

18:
  %19 = load ptr, ptr %2, align 8
  %20 = ptrtoint ptr %19 to i64
  %21 = sub i64 %20, ptrtoint (ptr @__typeid__ZTS1S_global_addr to i64)
  ; CHECK: zext {{.*}} @__typeid__ZTS1S_align
  %22 = zext nneg i8 ptrtoint (ptr @__typeid__ZTS1S_align to i8) to i64
  %23 = lshr i64 %21, %22
  ; CHECK: zext {{.*}} @__typeid__ZTS1S_align
  %24 = zext nneg i8 sub (i8 64, i8 ptrtoint (ptr @__typeid__ZTS1S_align to i8)) to i64
  %25 = shl i64 %21, %24
  %26 = or i64 %23, %25
  %27 = icmp ugt i64 %26, ptrtoint (ptr @__typeid__ZTS1S_size_m1 to i64)
  br i1 %27, label %28, label %29

28:
  tail call void @llvm.ubsantrap(i8 2) #5
  unreachable

29:
  %30 = load ptr, ptr %19, align 8
  tail call void %30(ptr noundef nonnull align 8 dereferenceable(8) %2)
  br label %31

31:
  %32 = phi i64 [ %24, %29 ], [ %11, %16 ]
  %33 = phi i64 [ %22, %29 ], [ %9, %16 ]
  %34 = load ptr, ptr %3, align 8
  %35 = ptrtoint ptr %34 to i64
  %36 = sub i64 %35, ptrtoint (ptr @__typeid__ZTS1S_global_addr to i64)
  ; CHECK: zext {{.*}} @__typeid__ZTS1S_align
  %37 = lshr i64 %36, %33
  ; CHECK: zext {{.*}} @__typeid__ZTS1S_align
  %38 = shl i64 %36, %32
  %39 = or i64 %37, %38
  %40 = icmp ugt i64 %39, ptrtoint (ptr @__typeid__ZTS1S_size_m1 to i64)
  br i1 %40, label %41, label %42

41:
  tail call void @llvm.ubsantrap(i8 2) #5
  unreachable

42:
  %43 = load ptr, ptr %34, align 8
  tail call void %43(ptr noundef nonnull align 8 dereferenceable(8) %3)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.ubsantrap(i8 immarg)

!0 = !{i64 0, i64 256}
!1 = !{i64 0, i64 128}
