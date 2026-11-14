; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -O0 -start-before=hexagon-lower-hlx-pair-allocas -stop-after=hexagon-lower-hlx-pair-allocas -o - %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: define{{.*}} @underaligned
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: store <1024 x i32> %value, ptr %pair, align 1024
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: %result = load <1024 x i32>, ptr %pair, align 1024
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: ret <1024 x i32> %result
define <1024 x i32> @underaligned(<1024 x i32> %value) {
entry:
  %pair = alloca <1024 x i32>, align 2048
  store <1024 x i32> %value, ptr %pair, align 1024
  %result = load <1024 x i32>, ptr %pair, align 1024
  ret <1024 x i32> %result
}

; CHECK-LABEL: define{{.*}} @derived_nonzero
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: %pair = getelementptr <1024 x i32>, ptr %pairs, i32 1
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: store <1024 x i32> %value, ptr %pair, align 2048
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: %result = load <1024 x i32>, ptr %pair, align 2048
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: ret <1024 x i32> %result
define <1024 x i32> @derived_nonzero(<1024 x i32> %value) {
entry:
  %pairs = alloca <1024 x i32>, i32 2, align 2048
  %pair = getelementptr <1024 x i32>, ptr %pairs, i32 1
  store <1024 x i32> %value, ptr %pair, align 2048
  %result = load <1024 x i32>, ptr %pair, align 2048
  ret <1024 x i32> %result
}

; CHECK-LABEL: define{{.*}} @volatile
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: store volatile <1024 x i32> %value, ptr %pair, align 2048
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: %result = load volatile <1024 x i32>, ptr %pair, align 2048
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: ret <1024 x i32> %result
define <1024 x i32> @volatile(<1024 x i32> %value) {
entry:
  %pair = alloca <1024 x i32>, align 2048
  store volatile <1024 x i32> %value, ptr %pair, align 2048
  %result = load volatile <1024 x i32>, ptr %pair, align 2048
  ret <1024 x i32> %result
}

; CHECK-LABEL: define{{.*}} @nontemporal
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: store <1024 x i32> %value, ptr %pair, align 2048, !nontemporal
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: %result = load <1024 x i32>, ptr %pair, align 2048, !nontemporal
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: ret <1024 x i32> %result
define <1024 x i32> @nontemporal(<1024 x i32> %value) {
entry:
  %pair = alloca <1024 x i32>, align 2048
  store <1024 x i32> %value, ptr %pair, align 2048, !nontemporal !0
  %result = load <1024 x i32>, ptr %pair, align 2048, !nontemporal !0
  ret <1024 x i32> %result
}

; AtomicExpand normally lowers these oversized atomic accesses to __atomic_*
; libcalls. Start at this pass to verify it preserves atomic IR if pass order
; changes.
; CHECK-LABEL: define{{.*}} @atomic
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: store atomic <1024 x i32> %value, ptr %pair seq_cst, align 2048
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: %result = load atomic <1024 x i32>, ptr %pair seq_cst, align 2048
; CHECK-NOT: @llvm.hexagon.LX.
; CHECK: ret <1024 x i32> %result
define <1024 x i32> @atomic(<1024 x i32> %value) {
entry:
  %pair = alloca <1024 x i32>, align 2048
  store atomic <1024 x i32> %value, ptr %pair seq_cst, align 2048
  %result = load atomic <1024 x i32>, ptr %pair seq_cst, align 2048
  ret <1024 x i32> %result
}

!0 = !{i32 1}
