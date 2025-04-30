; RUN: llc -mtriple=riscv64 -O3 -print-after=codegenprepare < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK-NO-GEP-OPT
; RUN: llc -mtriple=riscv64 -O3 -riscv-enable-gep-opt=true -print-after=codegenprepare < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK-GEP-OPT

define float @test_GEP(ptr %p, i64 %idxprom1) {
  %gep1 = getelementptr inbounds [3 x [3 x float]], ptr %p, i64 0, i64 %idxprom1, i32 1
  %load1 = load float, ptr %gep1, align 4
  %gep2 = getelementptr inbounds [3 x [3 x float]], ptr %p, i64 0, i64 %idxprom1, i32 3
  %load2 = load float, ptr %gep2, align 4
  %add = fadd float %load1, %load2
  ret float %add
}

; CHECK-NO-GEP-OPT: %gep1 = getelementptr inbounds [3 x [3 x float]], ptr %p, i64 0, i64 %idxprom1, i32 1
; CHECK-NO-GEP-OPT-NEXT: %load1 = load float, ptr %gep1, align 4
; CHECK-NO-GEP-OPT-NEXT: %gep2 = getelementptr inbounds [3 x [3 x float]], ptr %p, i64 0, i64 %idxprom1, i32 3
; CHECK-NO-GEP-OPT-NEXT: %load2 = load float, ptr %gep2, align 4

; CHECK-GEP-OPT: %1 = getelementptr [3 x [3 x float]], ptr %p, i64 0, i64 %idxprom1, i64 0
; CHECK-GEP-OPT-NEXT: %gep11 = getelementptr inbounds i8, ptr %1, i64 4
; CHECK-GEP-OPT-NEXT: %load1 = load float, ptr %gep11, align 4
; CHECK-GEP-OPT-NEXT: %gep22 = getelementptr inbounds i8, ptr %1, i64 12
; CHECK-GEP-OPT-NEXT: %load2 = load float, ptr %gep22, align 4
