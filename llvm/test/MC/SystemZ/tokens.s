# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid instruction
#CHECK: foo	100, 200
#CHECK: error: unknown token in expression
#CHECK: foo	100(, 200
#CHECK: error: invalid instruction
#CHECK: foo	100(200), 300
#CHECK: error: register expected
#CHECK: foo	100(200,), 300
#CHECK: error: invalid instruction
#CHECK: foo	100(200,%r1), 300
#CHECK: error: invalid address register
#CHECK: foo	100(%a0), 200
#CHECK: error: invalid instruction
#CHECK: foo	100(%r0), 200
#CHECK: error: invalid instruction
#CHECK: foo	100(%v0,%r1), 200
#CHECK: error: invalid instruction
#CHECK: foo	100(%v31), 200
#CHECK: error: invalid address register
#CHECK: foo	100(%r1,%a0), 200
#CHECK: error: unexpected token in address
#CHECK: foo	100(%r1,%r2, 200
#CHECK: error: invalid instruction
#CHECK: foo	100(%r1,%r2), 200
#CHECK: error: unexpected token in argument list
#CHECK: foo	100(%r1,%r2)(, 200
#CHECK: error: invalid instruction
#CHECK: foo	%r0, 200
#CHECK: error: invalid instruction
#CHECK: foo	%r15, 200
#CHECK: error: invalid register
#CHECK: foo	%r16, 200
#CHECK: error: invalid instruction
#CHECK: foo	%f0, 200
#CHECK: error: invalid instruction
#CHECK: foo	%f15, 200
#CHECK: error: invalid register
#CHECK: foo	%f16, 200
#CHECK: error: invalid instruction
#CHECK: foo	%a0, 200
#CHECK: error: invalid instruction
#CHECK: foo	%a15, 200
#CHECK: error: invalid register
#CHECK: foo	%a16, 200
#CHECK: error: invalid instruction
#CHECK: foo	%v0, 200
#CHECK: error: invalid instruction
#CHECK: foo	%v31, 200
#CHECK: error: invalid register
#CHECK: foo	%v32, 200
#CHECK: error: invalid register
#CHECK: foo	%c, 200
#CHECK: error: invalid register
#CHECK: foo	%, 200
#CHECK: error: unknown token in expression
#CHECK: foo	{, 200
#CHECK: error: invalid instruction
#CHECK: foo	100(15), 300
#CHECK: error: register expected
#CHECK: foo	100(15,), 300
#CHECK: error: invalid instruction
#CHECK: foo	100(15,%r1), 300
#CHECK: error: invalid instruction
#CHECK: foo	100(%v20,10), 300

	foo	100, 200
	foo	100(, 200
	foo	100(200), 300
	foo	100(200,), 300
	foo	100(200,%r1), 300
	foo	100(%a0), 200
	foo	100(%r0), 200
	foo	100(%v0,%r1), 200
	foo	100(%v31), 200
	foo	100(%r1,%a0), 200
	foo	100(%r1,%r2, 200
	foo	100(%r1,%r2), 200
	foo	100(%r1,%r2)(, 200
	foo	%r0, 200
	foo	%r15, 200
	foo	%r16, 200
	foo	%f0, 200
	foo	%f15, 200
	foo	%f16, 200
	foo	%a0, 200
	foo	%a15, 200
	foo	%a16, 200
	foo	%v0, 200
	foo	%v31, 200
	foo	%v32, 200
	foo	%c, 200
	foo	%, 200
	foo	{, 200
	foo	100(15), 300
	foo	100(15,), 300
	foo	100(15,%r1), 300
	foo	100(%v20,10), 300
