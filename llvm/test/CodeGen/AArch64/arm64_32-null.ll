; RUN: llc -fast-isel=true  -global-isel=false -O0 -mtriple=arm64_32-apple-ios %s -o - | FileCheck %s
; RUN: llc -fast-isel=false -global-isel=false -O0 -mtriple=arm64_32-apple-ios %s -o - | FileCheck %s

define void @test_store(ptr %p) {
; CHECK-LABEL: test_store:
; CHECK: mov [[R1:w[0-9]+]], wzr
; CHECK: str [[R1]], [x0]

  store ptr null, ptr %p
  ret void
}

define void @test_phi(ptr %p) {
; CHECK-LABEL: test_phi:
; CHECK: mov [[R1:x[0-9]+]], xzr
; CHECK: str [[R1]], [sp, #8]
; CHECK: b [[BB:LBB[0-9_]+]]
; CHECK: [[BB]]:
; CHECK: ldr x0, [sp, #8]
; CHECK: mov w8, w0
; CHECK: str w8, [x{{.*}}]

bb0:
  br label %bb1
bb1:
  %tmp0 = phi ptr [ null, %bb0 ]
  store ptr %tmp0, ptr %p
  ret void
}
