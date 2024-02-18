# RUN: llvm-mc -triple=armv7-linux-gnueabi %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple=armv7-linux-gnueabi --fdpic %s | llvm-readelf -h -r - | FileCheck %s

# RUN: not llvm-mc -filetype=obj -triple=armv7-linux-gnueabi %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ASM:      .long f(FUNCDESC)
# ASM-NEXT: .long f(GOTFUNCDESC)
# ASM-NEXT: .long f(GOTOFFFUNCDESC)

# CHECK:      OS/ABI: ARM FDPIC
# CHECK:      Machine: ARM
# CHECK:      Flags: 0x5000000

# CHECK:      R_ARM_FUNCDESC        00000000 f
# CHECK-NEXT: R_ARM_GOTFUNCDESC     00000000 f
# CHECK-NEXT: R_ARM_GOTOFFFUNCDESC  00000000 f
# CHECK-NEXT: R_ARM_TLS_GD32_FDPIC  00000000 tls
# CHECK-NEXT: R_ARM_TLS_LDM32_FDPIC 00000000 tls
# CHECK-NEXT: R_ARM_TLS_IE32_FDPIC  00000000 tls

.data
# ERR: [[#@LINE+1]]:7: error: relocation only supported in FDPIC mode
.long f(FUNCDESC)
# ERR: [[#@LINE+1]]:7: error: relocation only supported in FDPIC mode
.long f(GOTFUNCDESC)
# ERR: [[#@LINE+1]]:7: error: relocation only supported in FDPIC mode
.long f(GOTOFFFUNCDESC)
# ERR: [[#@LINE+1]]:7: error: relocation only supported in FDPIC mode
.long tls(tlsgd_fdpic)
.long tls(tlsldm_fdpic)
.long tls(gottpoff_fdpic)
