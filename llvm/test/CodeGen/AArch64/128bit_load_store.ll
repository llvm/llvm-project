; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=neon | FileCheck %s

define void @test_store_f128(ptr %ptr, fp128 %val) #0 {
; CHECK-LABEL: test_store_f128
; CHECK: str	 {{q[0-9]+}}, [{{x[0-9]+}}]
entry:
  store fp128 %val, ptr %ptr, align 16
  ret void
}

define fp128 @test_load_f128(ptr readonly %ptr) #2 {
; CHECK-LABEL: test_load_f128
; CHECK: ldr	 {{q[0-9]+}}, [{{x[0-9]+}}]
entry:
  %0 = load fp128, ptr %ptr, align 16
  ret fp128 %0
}

define void @test_vstrq_p128(ptr %ptr, i128 %val) #0 {
; CHECK-LABEL: test_vstrq_p128
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [{{x[0-9]+}}]

entry:
  %0 = bitcast i128 %val to fp128
  store fp128 %0, ptr %ptr, align 16
  ret void
}

define i128 @test_vldrq_p128(ptr readonly %ptr) #2 {
; CHECK-LABEL: test_vldrq_p128
; CHECK: ldp {{x[0-9]+}}, {{x[0-9]+}}, [{{x[0-9]+}}]

entry:
  %0 = load fp128, ptr %ptr, align 16
  %1 = bitcast fp128 %0 to i128
  ret i128 %1
}

define void @test_ld_st_p128(ptr nocapture %ptr) #0 {
; CHECK-LABEL: test_ld_st_p128
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}]
; CHECK-NEXT: str	{{q[0-9]+}}, [{{x[0-9]+}}, #16]
entry:
  %0 = load fp128, ptr %ptr, align 16
  %add.ptr = getelementptr inbounds i128, ptr %ptr, i64 1
  store fp128 %0, ptr %add.ptr, align 16
  ret void
}

