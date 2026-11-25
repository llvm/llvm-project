// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
// RUN: llvm-readelf -Srs %t | FileCheck %s

// test that we produce one relocation against each section.

# CHECK:      .foo      PROGBITS
# CHECK-NEXT: .foo      PROGBITS
# CHECK-NEXT: .bar      PROGBITS
# CHECK-NEXT: .rela.bar RELA

# CHECK:     Offset             Info   Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 0000000000000000  {{.*}} R_X86_64_64            0000000000000000 .foo + 0
# CHECK-NEXT: 0000000000000008  {{.*}} R_X86_64_64            0000000000000000 .foo + 0
# CHECK-NEXT: 0000000000000010  {{.*}} R_X86_64_64            0000000000000010 .rela.bar + 0
# CHECK-NEXT: 0000000000000018  {{.*}} R_X86_64_64            0000000000000000 .rela.und + 0

# CHECK:      0000000000000000     0 NOTYPE  LOCAL  DEFAULT [[#]] g
# CHECK-NEXT: 0000000000000010     0 NOTYPE  GLOBAL DEFAULT [[#]] .rela.bar
# CHECK-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT UND   .rela.und
# CHECK-EMPTY:

	.section	.foo,"aG",@progbits,v,comdat
f:

	.section	.foo,"a",@progbits
g:


	.section	.bar
	.quad	f
	.quad	g

.globl .rela.bar, .rela.und
.rela.bar:
  .quad .rela.bar
  .quad .rela.und
