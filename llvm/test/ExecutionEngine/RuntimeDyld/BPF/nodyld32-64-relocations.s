# RUN: llvm-mc -triple=bpfel -filetype=obj -o %t %s
# RUN: llvm-rtdyld -triple=bpfel -verify -check=%s %t

# test R_BPF_64_64 and R_BPF_64_NODYLD32 relocations, both should be ignored.

	.globl	_main
	.p2align	3
	.type	_main,@function
_main:                                  # @_main
	r1 = a ll

# rtdyld-check: decode_operand(_main, 1)[31:0] = 0x0

	r0 = *(u32 *)(r1 + 0)
	exit
.Lfunc_end0:
	.size	_main, .Lfunc_end0-_main
                                        # -- End function

	.type	a,@object                       # @a
	.section	.bss,"aw",@nobits
	.globl	a
	.p2align	2
a:
	.long	0                               # 0x0
	.size	a, 4

# rtdyld-check: *{4}a = 0

	.section	.BTF,"",@progbits
	.short	60319                           # 0xeb9f
	.byte	1
	.byte	0
	.long	24
	.long	0
	.long	80
	.long	80
	.long	87
	.long	0                               # BTF_KIND_FUNC_PROTO(id = 1)
	.long	218103808                       # 0xd000000
	.long	2
	.long	1                               # BTF_KIND_INT(id = 2)
	.long	16777216                        # 0x1000000
	.long	4
	.long	16777248                        # 0x1000020
	.long	5                               # BTF_KIND_FUNC(id = 3)
	.long	201326593                       # 0xc000001
	.long	1
	.long	80                              # BTF_KIND_VAR(id = 4)
	.long	234881024                       # 0xe000000
	.long	2
	.long	1
	.long	82                              # BTF_KIND_DATASEC(id = 5)
	.long	251658241                       # 0xf000001
	.long	0
	.long	4
btf_a:
	.long	a

# rtdyld-check: *{4}btf_a = 0

	.long	4
	.byte	0                               # string offset=0
	.ascii	"int"                           # string offset=1
	.byte	0
	.ascii	"_main"                         # string offset=5
	.byte	0
	.ascii	".text"                         # string offset=11
	.byte	0
	.ascii	"/home/yhs/work/tests/llvm/rtdyld/t.c" # string offset=17
	.byte	0
	.ascii	"int _main() { return a; }"     # string offset=54
	.byte	0
	.byte	97                              # string offset=80
	.byte	0
	.ascii	".bss"                          # string offset=82
	.byte	0
