# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# verify two extenders generated during relaxation
{
  if (p1) call foo_a
  if (!p1) call foo_b
}
# CHECK: 00004000 { immext(#0)
# CHECK: 5d004100   if (p1) call 0x0
# CHECK: 00004000   immext(#0)
# CHECK: 5d20c100   if (!p1) call 0x0 }

