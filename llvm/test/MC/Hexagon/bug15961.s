# RUN: llvm-mc -triple=hexagon -mv65 -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s
#

{
p0=cmp.eq(r18,#0)
if (p0.new) jumpr:nt r31
}

# The following had been getting duplexed to the :nt case
{
p0=cmp.eq(r18,#0)
if (p0.new) jumpr:t r31
}
#CHECK: 59a03fc6 { p0 = cmp.eq(r18,#0); if (p0.new) jumpr:nt r31 }
#CHECK: 75124000 { p0 = cmp.eq(r18,#0)
#CHECK: 535fd800   if (p0.new) jumpr:t r31 }

