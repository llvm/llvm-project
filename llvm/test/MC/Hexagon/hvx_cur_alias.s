# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -mhvx -filetype=obj %s 2>&1 | llvm-objdump --mattr=+hvx -d - | FileCheck %s

{ v2.cur = vmem(r11++m0)
  v5:4.h = vmpa(v3:2.ub,v5:4.ub)
}

# CHECK-NOT: warning: register `{{.+}}' used with `.cur' but not used in the same packet
# CHECK: vmpa
# CHECK: vmem
