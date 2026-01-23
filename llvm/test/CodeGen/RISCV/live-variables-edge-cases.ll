; RUN: llc -mtriple=riscv64 -riscv-enable-live-variables -verify-machineinstrs \
; RUN: -riscv-enable-live-variables -riscv-liveness-update-kills -stop-after=riscv-live-variables \
; RUN: -o - %s | FileCheck %s

; Test live variable analysis edge cases and special scenarios
; Including: dead code, unreachable blocks, critical edges, and complex phi nodes

; CHECK: test_dead_code
; CHECK:   bb.0.entry:
; CHECK:     liveins: $x10
; CHECK:     %0:gpr = COPY $x10
; CHECK:     $x10 = COPY killed %0
; CHECK:     PseudoRET implicit $x10

define i64 @test_dead_code(i64 %a, i64 %b) {
entry:
  %dead = add i64 %a, %b
  ret i64 %a
}

; CHECK: test_critical_edge
; CHECK:  bb.0.entry:
; CHECK:    successors: %bb.2(0x50000000), %bb.1(0x30000000)
; CHECK:    liveins: $x10, $x11, $x12
;
; CHECK:    %5:gpr = COPY $x12
; CHECK:    %4:gpr = COPY $x11
; CHECK:    %3:gpr = COPY $x10
; CHECK:    %6:gpr = COPY $x0
; CHECK:    BLT killed %6, %3, %bb.2
; CHECK:    PseudoBR %bb.1
;
; CHECK:  bb.1.check2:
; CHECK:    successors: %bb.2(0x50000000), %bb.3(0x30000000)
;
; CHECK:    %7:gpr = COPY $x0
; CHECK:    BGE killed %7, %4, %bb.3
; CHECK:    PseudoBR %bb.2
;
; CHECK:  bb.2.then:
; CHECK:    successors: %bb.4(0x80000000)
;
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    $x10 = COPY killed %3
; CHECK:    $x11 = COPY killed %4
; CHECK:    PseudoCALL target-flags(riscv-call) &__muldi3, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit $x11, implicit-def $x2, implicit-def $x10
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:    %8:gpr = COPY $x10
; CHECK:    %0:gpr = COPY killed %8
; CHECK:    PseudoBR %bb.4
;
; CHECK:  bb.3.else:
; CHECK:    successors: %bb.4(0x80000000)
;
; CHECK:    %1:gpr = SUB killed %3, killed %5
;
; CHECK:  bb.4.end:
; CHECK:    %2:gpr = PHI %1, %bb.3, %0, %bb.2
; CHECK:    $x10 = COPY killed %2
; CHECK:    PseudoRET implicit $x10

define i64 @test_critical_edge(i64 %a, i64 %b, i64 %c) {
entry:
  %cmp1 = icmp sgt i64 %a, 0
  br i1 %cmp1, label %then, label %check2

check2:
  %cmp2 = icmp sgt i64 %b, 0
  br i1 %cmp2, label %then, label %else

then:
  %mul = mul i64 %a, %b
  br label %end

else:
  %sub = sub i64 %a, %c
  br label %end

end:
  %result = phi i64 [ %mul, %then ], [ %sub, %else ]
  ret i64 %result
}

; CHECK: test_complex_phi
; CHECK:   bb.0.entry:
; CHECK:     successors: %bb.1(0x50000000), %bb.2(0x30000000)
; CHECK:     liveins: $x10, $x11, $x12, $x13
;
; CHECK:     %7:gpr = COPY $x13
; CHECK:     %6:gpr = COPY $x12
; CHECK:     %5:gpr = COPY $x11
; CHECK:     %4:gpr = COPY $x10
; CHECK:     %8:gpr = COPY $x0
; CHECK:     BGE killed %8, %4, %bb.2
; CHECK:     PseudoBR %bb.1
;
; CHECK:   bb.1.path1:
; CHECK:     successors: %bb.5(0x80000000)
;
; CHECK:     %0:gpr = ADD killed %4, killed %5
; CHECK:     PseudoBR %bb.5
;
; CHECK:   bb.2.path2:
; CHECK:     successors: %bb.3(0x50000000), %bb.4(0x30000000)
;
; CHECK:     %9:gpr = COPY $x0
; CHECK:     BGE killed %9, %6, %bb.4
; CHECK:     PseudoBR %bb.3
;
; CHECK:   bb.3.path2a:
; CHECK:     successors: %bb.5(0x80000000)
;
; CHECK:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:     $x10 = COPY killed %6
; CHECK:     $x11 = COPY killed %7
; CHECK:     PseudoCALL target-flags(riscv-call) &__muldi3, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit $x11, implicit-def $x2, implicit-def $x10
; CHECK:     ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:     %10:gpr = COPY $x10
; CHECK:     %1:gpr = COPY killed %10
; CHECK:     PseudoBR %bb.5
;
; CHECK:   bb.4.path2b:
; CHECK:     successors: %bb.5(0x80000000)
;
; CHECK:     %2:gpr = SUB killed %6, killed %7
;
; CHECK:   bb.5.merge:
; CHECK:     %3:gpr = PHI %2, %bb.4, %1, %bb.3, %0, %bb.1
; CHECK:     $x10 = COPY killed %3
; CHECK:     PseudoRET implicit $x10

define i64 @test_complex_phi(i64 %a, i64 %b, i64 %c, i64 %d) {
entry:
  %cmp1 = icmp sgt i64 %a, 0
  br i1 %cmp1, label %path1, label %path2

path1:
  %v1 = add i64 %a, %b
  br label %merge

path2:
  %cmp2 = icmp sgt i64 %c, 0
  br i1 %cmp2, label %path2a, label %path2b

path2a:
  %v2a = mul i64 %c, %d
  br label %merge

path2b:
  %v2b = sub i64 %c, %d
  br label %merge

merge:
  %result = phi i64 [ %v1, %path1 ], [ %v2a, %path2a ], [ %v2b, %path2b ]
  ret i64 %result
}

; CHECK: test_use_after_def
; CHECK:   bb.0.entry:
; CHECK:     liveins: $x10
;
; CHECK:     %0:gpr = COPY $x10
; CHECK:     %1:gpr = ADDI killed %0, 1
; CHECK:     %2:gpr = ADD killed %1, %1
; CHECK:     %3:gpr = ADDI killed %2, 5
; CHECK:     $x10 = COPY killed %3
; CHECK:     PseudoRET implicit $x10

define i64 @test_use_after_def(i64 %a) {
entry:
  %v1 = add i64 %a, 1
  %v2 = add i64 %v1, 2
  %v3 = add i64 %v2, 3
  %v4 = add i64 %v1, %v3
  ret i64 %v4
}

; CHECK: test_implicit_defs
; CHECK:   bb.0.entry:
; CHECK:     liveins: $x10, $x11
;
; CHECK:     %1:gpr = COPY $x11
; CHECK:     %0:gpr = COPY $x10
; CHECK:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:     $x10 = COPY %0
; CHECK:     $x11 = COPY %1
; CHECK:     PseudoCALL target-flags(riscv-call) &__divdi3, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit $x11, implicit-def $x2, implicit-def $x10
; CHECK:     ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:     %2:gpr = COPY $x10
; CHECK:     ADJCALLSTACKDOWN 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:     $x10 = COPY killed %0
; CHECK:     $x11 = COPY killed %1
; CHECK:     PseudoCALL target-flags(riscv-call) &__moddi3, csr_ilp32_lp64, implicit-def dead $x1, implicit $x10, implicit $x11, implicit-def $x2, implicit-def $x10
; CHECK:     ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
; CHECK:     %3:gpr = COPY $x10
; CHECK:     %4:gpr = ADD killed %2, killed %3
; CHECK:     $x10 = COPY killed %4
; CHECK:     PseudoRET implicit $x10

define i64 @test_implicit_defs(i64 %a, i64 %b) {
entry:
  %div = sdiv i64 %a, %b
  %rem = srem i64 %a, %b
  %sum = add i64 %div, %rem
  ret i64 %sum
}
