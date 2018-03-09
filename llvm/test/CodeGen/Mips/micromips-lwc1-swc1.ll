; RUN: llc -march=mips -mcpu=mips32r3 -mattr=+micromips \
; RUN:   -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MM32
; RUN: llc -march=mips -mcpu=mips32r6 -mattr=+micromips \
; RUN:   -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,MM32

@gf0 = external global float

define float @test_lwc1() {
entry:
; CHECK-LABEL: test_lwc1
; MM32:      lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; MM32:      addiu   $[[R1:[0-9]+]], $[[R0]], %lo(_gp_disp)
; MM32:      addu    $[[R2:[0-9]+]], $[[R1]], $25
; MM32:      lw      $[[R3:[0-9]+]], %got(gf0)($[[R2]])
; MM32:      lwc1    $f0, 0($[[R3]])

  %0 = load float, float* @gf0, align 4
  ret float %0
}

define void @test_swc1(float %a) {
entry:
; CHECK-LABEL: test_swc1
; MM32:      lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; MM32:      addiu   $[[R1:[0-9]+]], $[[R0]], %lo(_gp_disp)
; MM32:      addu    $[[R2:[0-9]+]], $[[R1]], $25
; MM32:      lw      $[[R3:[0-9]+]], %got(gf0)($[[R2]])
; MM32:      swc1    $f12, 0($[[R3]])

  store float %a, float* @gf0, align 4
  ret void
}

