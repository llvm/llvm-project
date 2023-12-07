; RUN: llc < %s -mtriple=aarch64-eabi -O=3 | FileCheck %s

define void @test() {
  ; CHECK: dmb sy
  call void @llvm.aarch64.dmb(i32 15)
  ; CHECK: dmb osh
  call void @llvm.aarch64.dmb(i32 3)
  ; CHECK: dsb sy
  call void @llvm.aarch64.dsb(i32 15)
  ; CHECK: dsb ishld
  call void @llvm.aarch64.dsb(i32 9)
  ; CHECK: isb
  call void @llvm.aarch64.isb(i32 15)
  ret void
}

; Important point is that the compiler should not reorder memory access
; instructions around DMB.
; Failure to do so, two STRs will collapse into one STP.
define void @test_dmb_reordering(i32 %a, i32 %b, ptr %d) {
  store i32 %a, ptr %d              ; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}]

  call void @llvm.aarch64.dmb(i32 15); CHECK: dmb sy

  %d1 = getelementptr i32, ptr %d, i64 1
  store i32 %b, ptr %d1             ; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}, #4]

  ret void
}

; Similarly for DSB.
define void @test_dsb_reordering(i32 %a, i32 %b, ptr %d) {
  store i32 %a, ptr %d              ; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}]

  call void @llvm.aarch64.dsb(i32 15); CHECK: dsb sy

  %d1 = getelementptr i32, ptr %d, i64 1
  store i32 %b, ptr %d1             ; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}, #4]

  ret void
}

; And ISB.
define void @test_isb_reordering(i32 %a, i32 %b, ptr %d) {
  store i32 %a, ptr %d              ; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}]

  call void @llvm.aarch64.isb(i32 15); CHECK: isb

  %d1 = getelementptr i32, ptr %d, i64 1
  store i32 %b, ptr %d1             ; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}, #4]

  ret void
}

declare void @llvm.aarch64.dmb(i32)
declare void @llvm.aarch64.dsb(i32)
declare void @llvm.aarch64.isb(i32)
