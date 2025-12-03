; RUN: opt < %s -mcpu=pwr9 -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s --check-prefix=P9
; RUN: opt < %s -mcpu=pwr10 -ppc-evl -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s --check-prefix=P10
; RUN: opt < %s -mcpu=future -ppc-evl -passes="print<cost-model>" 2>&1 -disable-output  | FileCheck %s --check-prefix=FUTURE
target datalayout = "e-m:e-Fn32-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

define void @bar(ptr %base, <2 x i8> %val) {
; P9: cost of 16 for {{.*}} @llvm.masked.load.v2i8.p0
; P9: cost of 12 for {{.*}} @llvm.masked.store.v2i8.p0
; P9: cost of 16 for {{.*}} @llvm.masked.load.v2i16.p0
; P9: cost of 12 for {{.*}} @llvm.masked.store.v2i16.p0
; P9: cost of 16 for {{.*}} @llvm.masked.load.v2i32.p0
; P9: cost of 12 for {{.*}} @llvm.masked.store.v2i32.p0
; P9: cost of 12 for {{.*}} @llvm.masked.load.v2i64.p0
; P9: cost of 10 for {{.*}} @llvm.masked.store.v2i64.p0
; P9: cost of 36 for {{.*}} @llvm.masked.load.v3i64.p0
; P9: cost of 15 for {{.*}} @llvm.masked.store.v3i64.p0
; P9: cost of 32 for {{.*}} @llvm.masked.load.v4i15.p0
; P9: cost of 24 for {{.*}} @llvm.masked.store.v4i15.p0
; P10: cost of 4 for {{.*}} @llvm.masked.load.v2i8.p0
; P10: cost of 4 for {{.*}} @llvm.masked.store.v2i8.p0
; P10: cost of 4 for {{.*}} @llvm.masked.load.v2i16.p0
; P10: cost of 4 for {{.*}} @llvm.masked.store.v2i16.p0
; P10: cost of 4 for {{.*}} @llvm.masked.load.v2i32.p0
; P10: cost of 4 for {{.*}} @llvm.masked.store.v2i32.p0
; P10: cost of 4 for {{.*}} @llvm.masked.load.v2i64.p0
; P10: cost of 4 for {{.*}} @llvm.masked.store.v2i64.p0
; P10: cost of 24 for {{.*}} @llvm.masked.load.v3i64.p0
; P10: cost of 12 for {{.*}} @llvm.masked.store.v3i64.p0
; P10: cost of 16 for {{.*}} @llvm.masked.load.v4i15.p0
; P10: cost of 16 for {{.*}} @llvm.masked.store.v4i15.p0
; FUTURE: cost of 3 for {{.*}} @llvm.masked.load.v2i8.p0
; FUTURE: cost of 3 for {{.*}} @llvm.masked.store.v2i8.p0
; FUTURE: cost of 4 for {{.*}} @llvm.masked.load.v2i16.p0
; FUTURE: cost of 4 for {{.*}} @llvm.masked.store.v2i16.p0
; FUTURE: cost of 4 for {{.*}} @llvm.masked.load.v2i32.p0
; FUTURE: cost of 4 for {{.*}} @llvm.masked.store.v2i32.p0
; FUTURE: cost of 4 for {{.*}} @llvm.masked.load.v2i64.p0
; FUTURE: cost of 4 for {{.*}} @llvm.masked.store.v2i64.p0
; FUTURE: cost of 24 for {{.*}} @llvm.masked.load.v3i64.p0
; FUTURE: cost of 12 for {{.*}} @llvm.masked.store.v3i64.p0
; FUTURE: cost of 16 for {{.*}} @llvm.masked.load.v4i15.p0
; FUTURE: cost of 16 for {{.*}} @llvm.masked.store.v4i15.p0
  %x1 = call <2 x i8> @llvm.masked.load.v2i8.p0(ptr %base, i32 1, <2 x i1> <i1 1, i1 1>, <2 x i8> %val)
  call void @llvm.masked.store.v2i8.p0(<2 x i8> %x1, ptr %base, i32 1, <2 x i1> <i1 1, i1 1>)
  %x2 = call <2 x i16> @llvm.masked.load.v2i16.p0(ptr %base, i32 1, <2 x i1> <i1 1, i1 1>, <2 x i16> <i16 0, i16 0>)
  call void @llvm.masked.store.v2i16.p0(<2 x i16> %x2, ptr %base, i32 1, <2 x i1> <i1 1, i1 1>)
  %x3 = call <2 x i32> @llvm.masked.load.v2i32.p0(ptr %base, i32 1, <2 x i1> <i1 1, i1 1>, <2 x i32> <i32 0, i32 0>)
  call void @llvm.masked.store.v2i32.p0(<2 x i32> %x3, ptr %base, i32 1, <2 x i1> <i1 1, i1 1>)
  %x4 = call <2 x i64> @llvm.masked.load.v2i64.p0(ptr %base, i32 1, <2 x i1> <i1 1, i1 1>, <2 x i64> <i64 0, i64 0>)
  call void @llvm.masked.store.v2i64.p0(<2 x i64> %x4, ptr %base, i32 1, <2 x i1> <i1 1, i1 1>)
  %x5 = call <3 x i64> @llvm.masked.load.v3i64.p0(ptr %base, i32 1, <3 x i1> <i1 1, i1 1, i1 1>, <3 x i64> <i64 0, i64 0, i64 0>)
  call void @llvm.masked.store.v3i64.p0(<3 x i64> %x5, ptr %base, i32 1, <3 x i1> <i1 1, i1 1, i1 1>)
  %x6 = call <4 x i15> @llvm.masked.load.v4i15.p0(ptr %base, i32 1, <4 x i1> <i1 1, i1 1, i1 1, i1 1>, <4 x i15> <i15 0, i15 0, i15 0, i15 0>)
  call void @llvm.masked.store.v4i15.p0(<4 x i15> %x6, ptr %base, i32 1, <4 x i1> <i1 1, i1 1, i1 1, i1 1>)
  ret void
}
