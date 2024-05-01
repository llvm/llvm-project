; Test long double atomic exchange.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr align 16 %ret, ptr align 16 %src, ptr align 16 %b) {
; CHECK-LABEL: f1:
; CHECK:       lg      %r1, 8(%r4)
; CHECK-NEXT:  lg      %r0, 0(%r4)
; CHECK-NEXT:  lg      %r4, 8(%r3)
; CHECK-NEXT:  lg      %r5, 0(%r3)
; CHECK-NEXT:.LBB0_1:                          # %atomicrmw.start
; CHECK-NEXT:                                  # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:  lgr     %r12, %r5
; CHECK-NEXT:  lgr     %r13, %r4
; CHECK-NEXT:  cdsg    %r12, %r0, 0(%r3)
; CHECK-NEXT:  lgr     %r4, %r13
; CHECK-NEXT:  lgr     %r5, %r12
; CHECK-NEXT:  jl      .LBB0_1
; CHECK-NEXT:# %bb.2:                          # %atomicrmw.end
; CHECK-NEXT:  stg     %r5, 0(%r2)
; CHECK-NEXT:  stg     %r4, 8(%r2)
; CHECK-NEXT:  lmg     %r12, %r15, 96(%r15)
; CHECK-NEXT:  br      %r14
  %val = load fp128, ptr %b, align 16
  %res = atomicrmw xchg ptr %src, fp128 %val seq_cst
  store fp128 %res, ptr %ret, align 16
  ret void
}

define void @f1_fpuse(ptr align 16 %ret, ptr align 16 %src, ptr align 16 %b) {
; CHECK-LABEL: f1_fpuse:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stmg %r12, %r15, 96(%r15)
; CHECK-NEXT:    .cfi_offset %r12, -64
; CHECK-NEXT:    .cfi_offset %r13, -56
; CHECK-NEXT:    .cfi_offset %r15, -40
; CHECK-NEXT:    aghi %r15, -176
; CHECK-NEXT:    .cfi_def_cfa_offset 336
; CHECK-NEXT:    ld %f0, 0(%r4)
; CHECK-NEXT:    ld %f2, 8(%r4)
; CHECK-NEXT:    lg %r0, 8(%r3)
; CHECK-NEXT:    lg %r1, 0(%r3)
; CHECK-NEXT:    axbr %f0, %f0
; CHECK-NEXT:    lgdr %r5, %f2
; CHECK-NEXT:    lgdr %r4, %f0
; CHECK-NEXT:  .LBB1_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    lgr %r12, %r1
; CHECK-NEXT:    lgr %r13, %r0
; CHECK-NEXT:    cdsg %r12, %r4, 0(%r3)
; CHECK-NEXT:    lgr %r0, %r13
; CHECK-NEXT:    lgr %r1, %r12
; CHECK-NEXT:    jl .LBB1_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    stg %r1, 160(%r15)
; CHECK-NEXT:    stg %r0, 168(%r15)
; CHECK-NEXT:    ld %f0, 160(%r15)
; CHECK-NEXT:    ld %f2, 168(%r15)
; CHECK-NEXT:    axbr %f0, %f0
; CHECK-NEXT:    std %f0, 0(%r2)
; CHECK-NEXT:    std %f2, 8(%r2)
; CHECK-NEXT:    lmg %r12, %r15, 272(%r15)
; CHECK-NEXT:    br %r14
  %val = load fp128, ptr %b, align 16
  %add.src = fadd fp128 %val, %val
  %res = atomicrmw xchg ptr %src, fp128 %add.src seq_cst
  %res.x2 = fadd fp128 %res, %res
  store fp128 %res.x2, ptr %ret, align 16
  ret void
}
