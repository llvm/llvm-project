* For z14 only.
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=z14 < %s 2> %t
* RUN: FileCheck < %t %s
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=arch12 < %s 2> %t
* RUN: FileCheck < %t %s

*CHECK: error: invalid instruction
	binle	0(1)

*CHECK: error: invalid instruction
	binhe	0(1)

*CHECK: error: invalid instruction
	bilh	0(1)

*CHECK: error: invalid instruction
	binlh	0(1)

*CHECK: error: invalid instruction
	bihe	0(1)

*CHECK: error: invalid instruction
	bile	0(1)
