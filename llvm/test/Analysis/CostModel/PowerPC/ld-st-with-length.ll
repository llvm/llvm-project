; RUN: opt < %s -mcpu=pwr9 -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s --check-prefix=P9
; RUN: opt < %s -mcpu=pwr10 -ppc-evl -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s --check-prefix=P10
; RUN: opt < %s -mcpu=future -ppc-evl -passes="print<cost-model>" 2>&1 -disable-output  | FileCheck %s --check-prefix=FUTURE
target datalayout = "e-m:e-Fn32-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

define void @bar(ptr %base, <2 x i8> %val) {
; P9: cost of 16 for {{.*}} @llvm.masked.load.v2i8.p0
; P10: cost of 4 for {{.*}} @llvm.masked.load.v2i8.p0
; FUTURE: cost of 3 for {{.*}} @llvm.masked.load.v2i8.p0
; P9: cost of 12 for {{.*}} @llvm.masked.store.v2i8.p0
; P10: cost of 4 for {{.*}} @llvm.masked.store.v2i8.p0
; FUTURE: cost of 3 for {{.*}} @llvm.masked.store.v2i8.p0
  %x2 = call <2 x i8> @llvm.masked.load.v2i8.p0(ptr %base, i32 1, <2 x i1> <i1 1, i1 1>, <2 x i8> %val)

  call void @llvm.masked.store.v2i8.p0(<2 x i8> %x2, ptr %base, i32 1, <2 x i1> <i1 1, i1 1>)

  ret void
}
