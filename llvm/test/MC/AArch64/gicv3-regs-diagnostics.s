// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2>&1 | FileCheck %s

        // Write-only
        mrs x10, icc_eoir1_el1
        mrs x7, icc_eoir0_el1
        mrs x22, icc_dir_el1
        mrs x24, icc_sgi1r_el1
        mrs x8, icc_asgi1r_el1
        mrs x28, icc_sgi0r_el1
// CHECK: error: expected readable system register
// CHECK-NEXT:         mrs x10, icc_eoir1_el1
// CHECK-NEXT:                  ^
// CHECK-NEXT: error: expected readable system register
// CHECK-NEXT:         mrs x7, icc_eoir0_el1
// CHECK-NEXT:                 ^
// CHECK-NEXT: error: expected readable system register
// CHECK-NEXT:         mrs x22, icc_dir_el1
// CHECK-NEXT:                  ^
// CHECK-NEXT: error: expected readable system register
// CHECK-NEXT:         mrs x24, icc_sgi1r_el1
// CHECK-NEXT:                  ^
// CHECK-NEXT: error: expected readable system register
// CHECK-NEXT:         mrs x8, icc_asgi1r_el1
// CHECK-NEXT:                 ^
// CHECK-NEXT: error: expected readable system register
// CHECK-NEXT:         mrs x28, icc_sgi0r_el1
// CHECK-NEXT:                  ^

        // Read-only
        msr icc_iar1_el1, x16
        msr icc_iar0_el1, x19
        msr icc_hppir1_el1, x29
        msr icc_hppir0_el1, x14
        msr icc_rpr_el1, x6
        msr ich_vtr_el2, x8
        msr ich_eisr_el2, x22
        msr ich_elsr_el2, x8
        msr ich_misr_el2, x10
// CHECK: error: expected writable system register or pstate
// CHECK-NEXT:         msr icc_iar1_el1, x16
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr icc_iar0_el1, x19
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr icc_hppir1_el1, x29
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr icc_hppir0_el1, x14
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr icc_rpr_el1, x6
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr ich_vtr_el2, x8
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr ich_eisr_el2, x22
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr ich_elsr_el2, x8
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr ich_misr_el2, x10
// CHECK-NEXT:             ^
