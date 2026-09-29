; RUN: llc < %s -mtriple=armv7 -mattr=+db | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7 -mattr=+db | FileCheck %s

; CHECK-LABEL: test
define void @test() {
  call void @llvm.arm.dmb(i32 3)     ; CHECK: dmb osh
  call void @llvm.arm.dsb(i32 7)     ; CHECK: dsb nsh
  call void @llvm.arm.isb(i32 15)    ; CHECK: isb sy
  ret void
}

; Important point is that the compiler should not reorder memory access
; instructions around DMB.
; Failure to do so, two STRs will collapse into one STRD.
; CHECK-LABEL: test_dmb_reordering
define void @test_dmb_reordering(i32 %a, i32 %b, ptr %d) {
  store i32 %a, ptr %d              ; CHECK: str {{r[0-9]+}}, [{{r[0-9]+}}]

  call void @llvm.arm.dmb(i32 15)    ; CHECK: dmb sy

  %d1 = getelementptr i32, ptr %d, i32 1
  store i32 %b, ptr %d1             ; CHECK: str {{r[0-9]+}}, [{{r[0-9]+}}, #4]

  ret void
}

; Similarly for DSB.
; CHECK-LABEL: test_dsb_reordering
define void @test_dsb_reordering(i32 %a, i32 %b, ptr %d) {
  store i32 %a, ptr %d              ; CHECK: str {{r[0-9]+}}, [{{r[0-9]+}}]

  call void @llvm.arm.dsb(i32 15)    ; CHECK: dsb sy

  %d1 = getelementptr i32, ptr %d, i32 1
  store i32 %b, ptr %d1             ; CHECK: str {{r[0-9]+}}, [{{r[0-9]+}}, #4]

  ret void
}

; And ISB.
; CHECK-LABEL: test_isb_reordering
define void @test_isb_reordering(i32 %a, i32 %b, ptr %d) {
  store i32 %a, ptr %d              ; CHECK: str {{r[0-9]+}}, [{{r[0-9]+}}]

  call void @llvm.arm.isb(i32 15)    ; CHECK: isb sy

  %d1 = getelementptr i32, ptr %d, i32 1
  store i32 %b, ptr %d1             ; CHECK: str {{r[0-9]+}}, [{{r[0-9]+}}, #4]

  ret void
}

declare void @llvm.arm.dmb(i32)
declare void @llvm.arm.dsb(i32)
declare void @llvm.arm.isb(i32)
