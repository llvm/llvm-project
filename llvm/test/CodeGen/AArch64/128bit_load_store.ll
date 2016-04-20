; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=neon | FileCheck %s

define void @test_store_f128(fp128* %ptr, fp128 %val) #0 {
; CHECK-LABEL: test_store_f128
; CHECK: str	 {{q[0-9]+}}, [{{x[0-9]+}}]
entry:
  store fp128 %val, fp128* %ptr, align 16
  ret void
}

define fp128 @test_load_f128(fp128* readonly %ptr) #2 {
; CHECK-LABEL: test_load_f128
; CHECK: ldr	 {{q[0-9]+}}, [{{x[0-9]+}}]
entry:
  %0 = load fp128, fp128* %ptr, align 16
  ret fp128 %0
}

define void @test_vstrq_p128(i128* %ptr, i128 %val) #0 {
; CHECK-LABEL: test_vstrq_p128
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [{{x[0-9]+}}]

entry:
  %0 = bitcast i128* %ptr to fp128*
  %1 = bitcast i128 %val to fp128
  store fp128 %1, fp128* %0, align 16
  ret void
}

define i128 @test_vldrq_p128(i128* readonly %ptr) #2 {
; CHECK-LABEL: test_vldrq_p128
; CHECK: ldp {{x[0-9]+}}, {{x[0-9]+}}, [{{x[0-9]+}}]

entry:
  %0 = bitcast i128* %ptr to fp128*
  %1 = load fp128, fp128* %0, align 16
  %2 = bitcast fp128 %1 to i128
  ret i128 %2
}

define void @test_ld_st_p128(i128* nocapture %ptr) #0 {
; CHECK-LABEL: test_ld_st_p128
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}]
; CHECK-NEXT: str	{{q[0-9]+}}, [{{x[0-9]+}}, #16]
entry:
  %0 = bitcast i128* %ptr to fp128*
  %1 = load fp128, fp128* %0, align 16
  %add.ptr = getelementptr inbounds i128, i128* %ptr, i64 1
  %2 = bitcast i128* %add.ptr to fp128*
  store fp128 %1, fp128* %2, align 16
  ret void
}

