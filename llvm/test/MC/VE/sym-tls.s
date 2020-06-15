# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

    lea %s0, x@tls_gd_lo(-24)
    and %s0, %s0, (32)0
    sic %s10
    lea.sl %s0, x@tls_gd_hi(%s10, %s0)
    lea %s12, __tls_get_addr@plt_lo(8)
    and %s12, %s12, (32)0
    lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
    bsic %s10, (, %s12)
# CHECK: lea %s0, x@tls_gd_lo(-24)
# CHECK-NEXT: and %s0, %s0, (32)0
# CHECK-NEXT: sic %s10
# CHECK-NEXT: lea.sl %s0, x@tls_gd_hi(%s10, %s0)
# CHECK-NEXT: lea %s12, __tls_get_addr@plt_lo(8)
# CHECK-NEXT: and %s12, %s12, (32)0
# CHECK-NEXT: lea.sl %s12, __tls_get_addr@plt_hi(%s10, %s12)
# CHECK-NEXT: bsic %s10, (, %s12)

# CHECK-OBJ: 0 R_VE_TLS_GD_LO32 x
# CHECK-OBJ-NEXT: 18 R_VE_TLS_GD_HI32 x
# CHECK-OBJ-NEXT: 20 R_VE_PLT_LO32 __tls_get_addr
# CHECK-OBJ-NEXT: 30 R_VE_PLT_HI32 __tls_get_addr
