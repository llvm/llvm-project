; RUN: llc -mtriple=riscv64 -riscv-enable-live-variables -verify-machineinstrs \
; RUN: -riscv-enable-live-variables -riscv-liveness-update-kills -stop-after=riscv-live-variables \
; RUN: -o - %s | FileCheck %s

; Test live variable analysis for RV64-specific scenarios
; This includes 64-bit operations, wide registers, and RV64-specific instructions

; CHECK:  test_64bit_ops
; CHECK:  bb.0.entry:
; CHECK:    liveins: $x10, $x11
;
; CHECK:    %1:gpr = COPY $x11
; CHECK:    %0:gpr = COPY $x10
; CHECK:    %2:gpr = SLLI killed %0, 32
; CHECK:    %3:gpr = OR killed %2, killed %1
; CHECK:    $x10 = COPY killed %3
; CHECK:    PseudoRET implicit $x10

define i64 @test_64bit_ops(i64 %a, i64 %b) {
entry:
  %shl = shl i64 %a, 32
  %or = or i64 %shl, %b
  ret i64 %or
}

; CHECK:  test_word_ops
; CHECK:  bb.0.entry:
; CHECK:    liveins: $x10, $x11
;
; CHECK:    %1:gpr = COPY $x11
; CHECK:    %0:gpr = COPY $x10
; CHECK:    %2:gpr = ADDW killed %0, killed %1
; CHECK:    $x10 = COPY killed %2
; CHECK:    PseudoRET implicit $x10

define i64 @test_word_ops(i64 %a, i64 %b) {
entry:
  %trunc_a = trunc i64 %a to i32
  %trunc_b = trunc i64 %b to i32
  %add = add i32 %trunc_a, %trunc_b
  %ext = sext i32 %add to i64
  ret i64 %ext
}

; CHECK:  test_mixed_width
; CHECK:  bb.0.entry:
; CHECK:    liveins: $x10, $x11
;
; CHECK:    %1:gpr = COPY $x11
; CHECK:    %0:gpr = COPY $x10
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    $x10 = COPY killed %0
; CHECK:    $x11 = COPY killed %1
; CHECK:    PseudoCALL target-flags(riscv-call) &__muldi3, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit $x11, implicit-def $x2, implicit-def $x10
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    %2:gpr = COPY $x10
; CHECK:    %3:gpr = ADDIW killed %2, 0
; CHECK:    $x10 = COPY killed %3
; CHECK:    PseudoRET implicit $x10

define i64 @test_mixed_width(i64 %a, i32 %b) {
entry:
  %ext_b = sext i32 %b to i64
  %mul = mul i64 %a, %ext_b
  %trunc = trunc i64 %mul to i32
  %final = sext i32 %trunc to i64
  ret i64 %final
}

; CHECK:  test_float_64
; CHECK:  bb.0.entry:
; CHECK:    successors: %bb.1(0x30000000), %bb.2(0x50000000)
; CHECK:    liveins: $x10, $x11, $x12
;
; CHECK:    %5:gpr = COPY $x12
; CHECK:    %4:gpr = COPY $x11
; CHECK:    %3:gpr = COPY $x10
; CHECK:    %7:gpr = COPY killed %4
; CHECK:    %6:gpr = COPY killed %3
; CHECK:    BNE killed %5, $x0, %bb.2
; CHECK:    PseudoBR %bb.1
;
; CHECK:  bb.1.then:
; CHECK:    successors: %bb.3(0x80000000)
;
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    $x10 = COPY killed %6
; CHECK:    $x11 = COPY killed %7
; CHECK:    PseudoCALL target-flags(riscv-call) &__adddf3, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit $x11, implicit-def $x2, implicit-def $x10
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    %9:gpr = COPY $x10
; CHECK:    %0:gpr = COPY killed %9
; CHECK:    PseudoBR %bb.3
;
; CHECK:  bb.2.else:
; CHECK:    successors: %bb.3(0x80000000)
;
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    $x10 = COPY killed %6
; CHECK:    $x11 = COPY killed %7
; CHECK:    PseudoCALL target-flags(riscv-call) &__muldf3, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit $x11, implicit-def $x2, implicit-def $x10
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    %8:gpr = COPY $x10
; CHECK:    %1:gpr = COPY killed %8
;
; CHECK:  bb.3.end:
; CHECK:    %2:gpr = PHI %1, %bb.2, %0, %bb.1
; CHECK:    $x10 = COPY killed %2
; CHECK:    PseudoRET implicit $x10

define double @test_float_64(double %a, double %b, i64 %selector) {
entry:
  %cmp = icmp eq i64 %selector, 0
  br i1 %cmp, label %then, label %else

then:
  %add = fadd double %a, %b
  br label %end

else:
  %mul = fmul double %a, %b
  br label %end

end:
  %result = phi double [ %add, %then ], [ %mul, %else ]
  ret double %result
}