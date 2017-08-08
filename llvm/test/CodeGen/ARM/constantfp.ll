; RUN: llc -mtriple=armv7 -mattr=+neon -mcpu=swift %s -o - | FileCheck %s
; RUN: llc -mtriple=armv7 -mattr=+neon -mcpu=cortex-a8 %s -o - | FileCheck --check-prefix=CHECK-NONEONFP %s
; RUN: llc -mtriple=armv7 -mattr=-neon -mcpu=cortex-a8 %s -o - | FileCheck --check-prefix=CHECK-NONEON %s

; RUN: llc -mtriple=thumbv7m -mcpu=cortex-m4 %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-NO-XO %s

; RUN: llc -mtriple=thumbv7m -mattr=+execute-only -mcpu=cortex-m4 %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-XO-FLOAT --check-prefix=CHECK-XO-DOUBLE %s

; RUN: llc -mtriple=thumbv7meb -mattr=+execute-only -mcpu=cortex-m4 %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-XO-FLOAT --check-prefix=CHECK-XO-DOUBLE-BE %s

; RUN: llc -mtriple=thumbv7m -mattr=+execute-only -mcpu=cortex-m4 -relocation-model=ropi %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-XO-ROPI %s

; RUN: llc -mtriple=thumbv8m.main -mattr=fp-armv8 %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-NO-XO %s

; RUN: llc -mtriple=thumbv8m.main -mattr=+execute-only -mattr=fp-armv8 %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-XO-FLOAT --check-prefix=CHECK-XO-DOUBLE %s

; RUN: llc -mtriple=thumbv8m.maineb -mattr=+execute-only -mattr=fp-armv8 %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-XO-FLOAT --check-prefix=CHECK-XO-DOUBLE-BE %s

; RUN: llc -mtriple=thumbv8m.main -mattr=+execute-only -mattr=fp-armv8 -relocation-model=ropi %s -o - \
; RUN: | FileCheck --check-prefix=CHECK-XO-ROPI %s

define arm_aapcs_vfpcc float @test_vmov_f32() {
; CHECK-LABEL: test_vmov_f32:
; CHECK: vmov.f32 d0, #1.0

; CHECK-NONEONFP: vmov.f32 s0, #1.0
  ret float 1.0
}

define arm_aapcs_vfpcc float @test_vmov_imm() {
; CHECK-LABEL: test_vmov_imm:
; CHECK: vmov.i32 d0, #0

; CHECK-NONEON-LABEL: test_vmov_imm:
; CHECK-NONEON: vldr s0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-NO-XO-LABEL: test_vmov_imm:
; CHECK-NO-XO: vldr s0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-XO-FLOAT-LABEL: test_vmov_imm:
; CHECK-XO-FLOAT: movs [[REG:r[0-9]+]], #0
; CHECK-XO-FLOAT: vmov {{s[0-9]+}}, [[REG]]
; CHECK-XO-FLOAT-NOT: vldr
  ret float 0.0
}

define arm_aapcs_vfpcc float @test_vmvn_imm() {
; CHECK-LABEL: test_vmvn_imm:
; CHECK: vmvn.i32 d0, #0xb0000000

; CHECK-NONEON-LABEL: test_vmvn_imm:
; CHECK-NONEON: vldr s0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-NO-XO-LABEL: test_vmvn_imm:
; CHECK-NO-XO: vldr s0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-XO-FLOAT-LABEL: test_vmvn_imm:
; CHECK-XO-FLOAT: mvn [[REG:r[0-9]+]], #-1342177280
; CHECK-XO-FLOAT: vmov {{s[0-9]+}}, [[REG]]
; CHECK-XO-FLOAT-NOT: vldr
  ret float 8589934080.0
}

define arm_aapcs_vfpcc double @test_vmov_f64() {
; CHECK-LABEL: test_vmov_f64:
; CHECK: vmov.f64 d0, #1.0

; CHECK-NONEON-LABEL: test_vmov_f64:
; CHECK-NONEON: vmov.f64 d0, #1.0

  ret double 1.0
}

define arm_aapcs_vfpcc double @test_vmov_double_imm() {
; CHECK-LABEL: test_vmov_double_imm:
; CHECK: vmov.i32 d0, #0

; CHECK-NONEON-LABEL: test_vmov_double_imm:
; CHECK-NONEON: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-NO-XO-LABEL: test_vmov_double_imm:
; CHECK-NO-XO: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-XO-DOUBLE-LABEL: test_vmov_double_imm:
; CHECK-XO-DOUBLE: movs [[REG:r[0-9]+]], #0
; CHECK-XO-DOUBLE: vmov {{d[0-9]+}}, [[REG]], [[REG]]
; CHECK-XO-DOUBLE-NOT: vldr

; CHECK-XO-DOUBLE-BE-LABEL: test_vmov_double_imm:
; CHECK-XO-DOUBLE-BE: movs [[REG:r[0-9]+]], #0
; CHECK-XO-DOUBLE-BE: vmov {{d[0-9]+}}, [[REG]], [[REG]]
; CHECK-XO-DOUBLE-NOT: vldr
  ret double 0.0
}

define arm_aapcs_vfpcc double @test_vmvn_double_imm() {
; CHECK-LABEL: test_vmvn_double_imm:
; CHECK: vmvn.i32 d0, #0xb0000000

; CHECK-NONEON-LABEL: test_vmvn_double_imm:
; CHECK-NONEON: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-NO-XO-LABEL: test_vmvn_double_imm:
; CHECK-NO-XO: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-XO-DOUBLE-LABEL: test_vmvn_double_imm:
; CHECK-XO-DOUBLE: mvn [[REG:r[0-9]+]], #-1342177280
; CHECK-XO-DOUBLE: vmov {{d[0-9]+}}, [[REG]], [[REG]]
; CHECK-XO-DOUBLE-NOT: vldr

; CHECK-XO-DOUBLE-BE-LABEL: test_vmvn_double_imm:
; CHECK-XO-DOUBLE-BE: mvn [[REG:r[0-9]+]], #-1342177280
; CHECK-XO-DOUBLE-BE: vmov {{d[0-9]+}}, [[REG]], [[REG]]
; CHECK-XO-DOUBLE-BE-NOT: vldr
  ret double 0x4fffffff4fffffff
}

; Make sure we don't ignore the high half of 64-bit values when deciding whether
; a vmov/vmvn is possible.
define arm_aapcs_vfpcc double @test_notvmvn_double_imm() {
; CHECK-LABEL: test_notvmvn_double_imm:
; CHECK: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-NONEON-LABEL: test_notvmvn_double_imm:
; CHECK-NONEON: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-NO-XO-LABEL: test_notvmvn_double_imm:
; CHECK-NO-XO: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-XO-DOUBLE-LABEL: test_notvmvn_double_imm:
; CHECK-XO-DOUBLE: mvn [[REG1:r[0-9]+]], #-1342177280
; CHECK-XO-DOUBLE: mov.w [[REG2:r[0-9]+]], #-1
; CHECK-XO-DOUBLE: vmov {{d[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-XO-DOUBLE-NOT: vldr

; CHECK-XO-DOUBLE-BE-LABEL: test_notvmvn_double_imm:
; CHECK-XO-DOUBLE-BE: mov.w [[REG1:r[0-9]+]], #-1
; CHECK-XO-DOUBLE-BE: mvn [[REG2:r[0-9]+]], #-1342177280
; CHECK-XO-DOUBLE-BE: vmov {{d[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-XO-DOUBLE-BE-NOT: vldr
  ret double 0x4fffffffffffffff
}

define arm_aapcs_vfpcc float @lower_const_f32_xo() {
; CHECK-NO-XO-LABEL: lower_const_f32_xo
; CHECK-NO-XO: vldr {{s[0-9]+}}, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-XO-FLOAT-LABEL: lower_const_f32_xo
; CHECK-XO-FLOAT: movw [[REG:r[0-9]+]], #29884
; CHECK-XO-FLOAT: movt [[REG]], #16083
; CHECK-XO-FLOAT: vmov {{s[0-9]+}}, [[REG]]
; CHECK-XO-FLOAT-NOT: vldr
  ret float 0x3FDA6E9780000000
}

define arm_aapcs_vfpcc double @lower_const_f64_xo() {
; CHECK-NO-XO-LABEL: lower_const_f64_xo
; CHECK-NO-XO: vldr {{d[0-9]+}}, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-XO-DOUBLE-LABEL: lower_const_f64_xo
; CHECK-XO-DOUBLE: movw [[REG1:r[0-9]+]], #6291
; CHECK-XO-DOUBLE: movw [[REG2:r[0-9]+]], #27263
; CHECK-XO-DOUBLE: movt [[REG1]], #16340
; CHECK-XO-DOUBLE: movt [[REG2]], #29884
; CHECK-XO-DOUBLE: vmov {{d[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-XO-DOUBLE-NOT: vldr

; CHECK-XO-DOUBLE-BE-LABEL: lower_const_f64_xo
; CHECK-XO-DOUBLE-BE: movw [[REG1:r[0-9]+]], #27263
; CHECK-XO-DOUBLE-BE: movw [[REG2:r[0-9]+]], #6291
; CHECK-XO-DOUBLE-BE: movt [[REG1]], #29884
; CHECK-XO-DOUBLE-BE: movt [[REG2]], #16340
; CHECK-XO-DOUBLE-BE: vmov {{d[0-9]+}}, [[REG2]], [[REG1]]
; CHECK-XO-DOUBLE-BE-NOT: vldr
  ret double 3.140000e-01
}

; This is a target independent optimization, performed by the
; DAG Combiner, which promotes floating point literals into
; constant pools:
;
; (a cond b) ? 1.0f : 2.0f -> load (ConstPoolAddr + ((a cond b) ? 0 : 4)
;
; We need to make sure that the constant pools are placed in
; the data section when generating execute-only code:

define arm_aapcs_vfpcc float @lower_fpconst_select(float %f) {

; CHECK-NO-XO-LABEL: lower_fpconst_select
; CHECK-NO-XO: adr [[REG:r[0-9]+]], [[LABEL:.?LCPI[0-9]+_[0-9]+]]
; CHECK-NO-XO: vldr {{s[0-9]+}}, {{[[]}}[[REG]]{{[]]}}
; CHECK-NO-XO-NOT: .rodata
; CHECK-NO-XO: [[LABEL]]:
; CHECK-NO-XO: .long   1335165689
; CHECK-NO-XO: .long   1307470632

; CHECK-XO-FLOAT-LABEL: lower_fpconst_select
; CHECK-XO-FLOAT: movw [[REG:r[0-9]+]], :lower16:[[LABEL:.?LCP[0-9]+_[0-9]+]]
; CHECK-XO-FLOAT: movt [[REG]], :upper16:[[LABEL]]
; CHECK-XO-FLOAT: vldr {{s[0-9]+}}, {{[[]}}[[REG]]{{[]]}}
; CHECK-XO-FLOAT: .rodata
; CHECK-XO-FLOAT-NOT: .text
; CHECK-XO-FLOAT: [[LABEL]]:
; CHECK-XO-FLOAT: .long   1335165689
; CHECK-XO-FLOAT: .long   1307470632

; CHECK-XO-ROPI-LABEL: lower_fpconst_select
; CHECK-XO-ROPI: movw [[REG:r[0-9]+]], :lower16:([[LABEL1:.?LCP[0-9]+_[0-9]+]]-([[LABEL2:.?LPC[0-9]+_[0-9]+]]+4))
; CHECK-XO-ROPI: movt [[REG]], :upper16:([[LABEL1]]-([[LABEL2]]+4))
; CHECK-XO-ROPI: [[LABEL2]]:
; CHECK-XO-ROPI: vldr {{s[0-9]+}}, {{[[]}}[[REG]]{{[]]}}
; CHECK-XO-ROPI: .rodata
; CHECK-XO-ROPI-NOT: .text
; CHECK-XO-ROPI: [[LABEL1]]:
; CHECK-XO-ROPI: .long   1335165689
; CHECK-XO-ROPI: .long   1307470632

  %cmp = fcmp nnan oeq float %f, 0.000000e+00
  %sel = select i1 %cmp, float 5.000000e+08, float 5.000000e+09
  ret float %sel
}
