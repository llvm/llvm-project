; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr8 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck  %s

; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr8 -disable-ppc-vsx-swap-removal \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck  \
; RUN:   -check-prefix=NOOPTSWAP %s

; RUN: llc -O3 -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:  -verify-machineinstrs -ppc-vsr-nums-as-vr < %s | FileCheck  \
; RUN:  -check-prefix=CHECK-P9 --implicit-check-not xxswapd %s

; RUN: llc -O3 -mcpu=pwr9 -disable-ppc-vsx-swap-removal -mattr=-power9-vector \
; RUN:  -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:  | FileCheck  -check-prefix=NOOPTSWAP %s

; LH: 2016-11-17
;   Updated align attritue from 16 to 8 to keep swap instructions tests.
;   Changes have been made on little-endian to use lvx and stvx
;   instructions instead of lxvd2x/xxswapd and xxswapd/stxvd2x for
;   aligned vectors with elements up to 4 bytes

; This test was generated from the following source:
;
; #define N 4096
; int ca[N] __attribute__((aligned(16)));
; int cb[N] __attribute__((aligned(16)));
; int cc[N] __attribute__((aligned(16)));
; int cd[N] __attribute__((aligned(16)));
;
; void foo ()
; {
;   int i;
;   for (i = 0; i < N; i++) {
;     ca[i] = (cb[i] + cc[i]) * cd[i];
;   }
; }

@cb = common global [4096 x i32] zeroinitializer, align 8
@cc = common global [4096 x i32] zeroinitializer, align 8
@cd = common global [4096 x i32] zeroinitializer, align 8
@ca = common global [4096 x i32] zeroinitializer, align 8

define void @foo() {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next.3, %vector.body ]
  %0 = getelementptr inbounds [4096 x i32], ptr @cb, i64 0, i64 %index
  %wide.load = load <4 x i32>, ptr %0, align 8
  %1 = getelementptr inbounds [4096 x i32], ptr @cc, i64 0, i64 %index
  %wide.load13 = load <4 x i32>, ptr %1, align 8
  %2 = add nsw <4 x i32> %wide.load13, %wide.load
  %3 = getelementptr inbounds [4096 x i32], ptr @cd, i64 0, i64 %index
  %wide.load14 = load <4 x i32>, ptr %3, align 8
  %4 = mul nsw <4 x i32> %2, %wide.load14
  %5 = getelementptr inbounds [4096 x i32], ptr @ca, i64 0, i64 %index
  store <4 x i32> %4, ptr %5, align 8
  %index.next = add nuw nsw i64 %index, 4
  %6 = getelementptr inbounds [4096 x i32], ptr @cb, i64 0, i64 %index.next
  %wide.load.1 = load <4 x i32>, ptr %6, align 8
  %7 = getelementptr inbounds [4096 x i32], ptr @cc, i64 0, i64 %index.next
  %wide.load13.1 = load <4 x i32>, ptr %7, align 8
  %8 = add nsw <4 x i32> %wide.load13.1, %wide.load.1
  %9 = getelementptr inbounds [4096 x i32], ptr @cd, i64 0, i64 %index.next
  %wide.load14.1 = load <4 x i32>, ptr %9, align 8
  %10 = mul nsw <4 x i32> %8, %wide.load14.1
  %11 = getelementptr inbounds [4096 x i32], ptr @ca, i64 0, i64 %index.next
  store <4 x i32> %10, ptr %11, align 8
  %index.next.1 = add nuw nsw i64 %index.next, 4
  %12 = getelementptr inbounds [4096 x i32], ptr @cb, i64 0, i64 %index.next.1
  %wide.load.2 = load <4 x i32>, ptr %12, align 8
  %13 = getelementptr inbounds [4096 x i32], ptr @cc, i64 0, i64 %index.next.1
  %wide.load13.2 = load <4 x i32>, ptr %13, align 8
  %14 = add nsw <4 x i32> %wide.load13.2, %wide.load.2
  %15 = getelementptr inbounds [4096 x i32], ptr @cd, i64 0, i64 %index.next.1
  %wide.load14.2 = load <4 x i32>, ptr %15, align 8
  %16 = mul nsw <4 x i32> %14, %wide.load14.2
  %17 = getelementptr inbounds [4096 x i32], ptr @ca, i64 0, i64 %index.next.1
  store <4 x i32> %16, ptr %17, align 8
  %index.next.2 = add nuw nsw i64 %index.next.1, 4
  %18 = getelementptr inbounds [4096 x i32], ptr @cb, i64 0, i64 %index.next.2
  %wide.load.3 = load <4 x i32>, ptr %18, align 8
  %19 = getelementptr inbounds [4096 x i32], ptr @cc, i64 0, i64 %index.next.2
  %wide.load13.3 = load <4 x i32>, ptr %19, align 8
  %20 = add nsw <4 x i32> %wide.load13.3, %wide.load.3
  %21 = getelementptr inbounds [4096 x i32], ptr @cd, i64 0, i64 %index.next.2
  %wide.load14.3 = load <4 x i32>, ptr %21, align 8
  %22 = mul nsw <4 x i32> %20, %wide.load14.3
  %23 = getelementptr inbounds [4096 x i32], ptr @ca, i64 0, i64 %index.next.2
  store <4 x i32> %22, ptr %23, align 8
  %index.next.3 = add nuw nsw i64 %index.next.2, 4
  %24 = icmp eq i64 %index.next.3, 4096
  br i1 %24, label %for.end, label %vector.body

for.end:
  ret void
}

; CHECK-LABEL: @foo
; CHECK-NOT: xxpermdi
; CHECK-NOT: xxswapd
; CHECK-P9-NOT: xxpermdi

; CHECK: lxvd2x
; CHECK: lxvd2x
; CHECK-DAG: lxvd2x
; CHECK-DAG: vadduwm
; CHECK: vmuluwm
; CHECK: stxvd2x

; CHECK: lxvd2x
; CHECK: lxvd2x
; CHECK-DAG: lxvd2x
; CHECK-DAG: vadduwm
; CHECK: vmuluwm
; CHECK: stxvd2x

; CHECK: lxvd2x
; CHECK: lxvd2x
; CHECK-DAG: lxvd2x
; CHECK-DAG: vadduwm
; CHECK: vmuluwm
; CHECK: stxvd2x

; CHECK: lxvd2x
; CHECK: lxvd2x
; CHECK-DAG: lxvd2x
; CHECK-DAG: vadduwm
; CHECK: vmuluwm
; CHECK: stxvd2x

; NOOPTSWAP-LABEL: @foo

; NOOPTSWAP: lxvd2x
; NOOPTSWAP-DAG: lxvd2x
; NOOPTSWAP-DAG: lxvd2x
; NOOPTSWAP-DAG: xxswapd
; NOOPTSWAP-DAG: xxswapd
; NOOPTSWAP-DAG: xxswapd
; NOOPTSWAP-DAG: vadduwm
; NOOPTSWAP: vmuluwm
; NOOPTSWAP: xxswapd
; NOOPTSWAP-DAG: xxswapd
; NOOPTSWAP-DAG: xxswapd
; NOOPTSWAP-DAG: stxvd2x
; NOOPTSWAP-DAG: stxvd2x
; NOOPTSWAP: stxvd2x

; CHECK-P9-LABEL: @foo
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: lxv
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: stxv
; CHECK-P9-DAG: stxv
; CHECK-P9-DAG: stxv
; CHECK-P9-DAG: stxv

