; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr8 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr8 -disable-ppc-vsx-swap-removal \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck \
; RUN:   -check-prefix=NOOPTSWAP %s

; RUN: llc -O3 -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:  -verify-machineinstrs -ppc-vsr-nums-as-vr < %s | FileCheck \
; RUN:  -check-prefix=CHECK-P9 --implicit-check-not xxswapd %s

; RUN: llc -O3 -mcpu=pwr9 -disable-ppc-vsx-swap-removal -mattr=-power9-vector \
; RUN:  -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:  | FileCheck -check-prefix=NOOPTSWAP %s

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

@cb = common global [4096 x i32] zeroinitializer, align 16
@cc = common global [4096 x i32] zeroinitializer, align 16
@cd = common global [4096 x i32] zeroinitializer, align 16
@ca = common global [4096 x i32] zeroinitializer, align 16

define void @foo() {
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next.3, %vector.body ]
  %0 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cb, i64 0, i64 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 16
  %2 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cc, i64 0, i64 %index
  %3 = bitcast i32* %2 to <4 x i32>*
  %wide.load13 = load <4 x i32>, <4 x i32>* %3, align 16
  %4 = add nsw <4 x i32> %wide.load13, %wide.load
  %5 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cd, i64 0, i64 %index
  %6 = bitcast i32* %5 to <4 x i32>*
  %wide.load14 = load <4 x i32>, <4 x i32>* %6, align 16
  %7 = mul nsw <4 x i32> %4, %wide.load14
  %8 = getelementptr inbounds [4096 x i32], [4096 x i32]* @ca, i64 0, i64 %index
  %9 = bitcast i32* %8 to <4 x i32>*
  store <4 x i32> %7, <4 x i32>* %9, align 16
  %index.next = add nuw nsw i64 %index, 4
  %10 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cb, i64 0, i64 %index.next
  %11 = bitcast i32* %10 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %11, align 16
  %12 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cc, i64 0, i64 %index.next
  %13 = bitcast i32* %12 to <4 x i32>*
  %wide.load13.1 = load <4 x i32>, <4 x i32>* %13, align 16
  %14 = add nsw <4 x i32> %wide.load13.1, %wide.load.1
  %15 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cd, i64 0, i64 %index.next
  %16 = bitcast i32* %15 to <4 x i32>*
  %wide.load14.1 = load <4 x i32>, <4 x i32>* %16, align 16
  %17 = mul nsw <4 x i32> %14, %wide.load14.1
  %18 = getelementptr inbounds [4096 x i32], [4096 x i32]* @ca, i64 0, i64 %index.next
  %19 = bitcast i32* %18 to <4 x i32>*
  store <4 x i32> %17, <4 x i32>* %19, align 16
  %index.next.1 = add nuw nsw i64 %index.next, 4
  %20 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cb, i64 0, i64 %index.next.1
  %21 = bitcast i32* %20 to <4 x i32>*
  %wide.load.2 = load <4 x i32>, <4 x i32>* %21, align 16
  %22 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cc, i64 0, i64 %index.next.1
  %23 = bitcast i32* %22 to <4 x i32>*
  %wide.load13.2 = load <4 x i32>, <4 x i32>* %23, align 16
  %24 = add nsw <4 x i32> %wide.load13.2, %wide.load.2
  %25 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cd, i64 0, i64 %index.next.1
  %26 = bitcast i32* %25 to <4 x i32>*
  %wide.load14.2 = load <4 x i32>, <4 x i32>* %26, align 16
  %27 = mul nsw <4 x i32> %24, %wide.load14.2
  %28 = getelementptr inbounds [4096 x i32], [4096 x i32]* @ca, i64 0, i64 %index.next.1
  %29 = bitcast i32* %28 to <4 x i32>*
  store <4 x i32> %27, <4 x i32>* %29, align 16
  %index.next.2 = add nuw nsw i64 %index.next.1, 4
  %30 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cb, i64 0, i64 %index.next.2
  %31 = bitcast i32* %30 to <4 x i32>*
  %wide.load.3 = load <4 x i32>, <4 x i32>* %31, align 16
  %32 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cc, i64 0, i64 %index.next.2
  %33 = bitcast i32* %32 to <4 x i32>*
  %wide.load13.3 = load <4 x i32>, <4 x i32>* %33, align 16
  %34 = add nsw <4 x i32> %wide.load13.3, %wide.load.3
  %35 = getelementptr inbounds [4096 x i32], [4096 x i32]* @cd, i64 0, i64 %index.next.2
  %36 = bitcast i32* %35 to <4 x i32>*
  %wide.load14.3 = load <4 x i32>, <4 x i32>* %36, align 16
  %37 = mul nsw <4 x i32> %34, %wide.load14.3
  %38 = getelementptr inbounds [4096 x i32], [4096 x i32]* @ca, i64 0, i64 %index.next.2
  %39 = bitcast i32* %38 to <4 x i32>*
  store <4 x i32> %37, <4 x i32>* %39, align 16
  %index.next.3 = add nuw nsw i64 %index.next.2, 4
  %40 = icmp eq i64 %index.next.3, 4096
  br i1 %40, label %for.end, label %vector.body

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
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: lxvx
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vadduwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: vmuluwm
; CHECK-P9-DAG: stxvx
; CHECK-P9-DAG: stxvx
; CHECK-P9-DAG: stxvx
; CHECK-P9-DAG: stxvx

