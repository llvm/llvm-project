# REQUIRES: x86_64-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -filetype=obj -o %t/tls.o %s
# RUN: llvm-rtdyld -triple=x86_64-unknown-linux -execute %t/tls.o


_main:

	push %rbx
	# load the address of the GOT in rbx for the large code model tests
	lea _GLOBAL_OFFSET_TABLE_(%rip), %rbx

# Test Local Exec TLS Model
	mov %fs:tls_foo@tpoff, %eax
	cmp $0x12, %eax
	je 1f
	mov $1, %eax
	jmp 2f
1:

	mov %fs:tls_bar@tpoff, %eax
	cmp $0x34, %eax
	je 1f
	mov $2, %eax
	jmp 2f
1:

# Test Initial Exec TLS Model
	mov tls_foo@gottpoff(%rip), %rax
	mov %fs:(%rax), %eax
	cmp $0x12, %eax
	je 1f
	mov $3, %eax
	jmp 2f
1:

	mov tls_bar@gottpoff(%rip), %rax
	mov %fs:(%rax), %eax
	cmp $0x34, %eax
	je 1f
	mov $4, %eax
	jmp 2f
1:

# Test Local Dynamic TLS Model (small code model)
	lea tls_foo@tlsld(%rip), %rdi
	call __tls_get_addr@plt
	mov tls_foo@dtpoff(%rax), %eax
	cmp $0x12, %eax
	je 1f
	mov $5, %eax
	jmp 2f
1:

	lea tls_bar@tlsld(%rip), %rdi
	call __tls_get_addr@plt
	mov tls_bar@dtpoff(%rax), %eax
	cmp $0x34, %eax
	je 1f
	mov $6, %eax
	jmp 2f
1:

# Test Local Dynamic TLS Model (large code model)
	lea tls_foo@tlsld(%rip), %rdi
	movabs $__tls_get_addr@pltoff, %rax
	add %rbx, %rax
	call *%rax
	mov tls_foo@dtpoff(%rax), %eax
	cmp $0x12, %eax
	je 1f
	mov $7, %eax
	jmp 2f
1:

	lea tls_bar@tlsld(%rip), %rdi
	movabs $__tls_get_addr@pltoff, %rax
	add %rbx, %rax
	call *%rax
	mov tls_bar@dtpoff(%rax), %eax
	cmp $0x34, %eax
	je 1f
	mov $8, %eax
	jmp 2f
1:

# Test Global Dynamic TLS Model (small code model)
	.byte 0x66
	leaq tls_foo@tlsgd(%rip), %rdi
	.byte 0x66, 0x66, 0x48
	call __tls_get_addr@plt
	mov (%rax), %eax
	cmp $0x12, %eax
	je 1f
	mov $9, %eax
	jmp 2f
1:

	.byte 0x66
	leaq tls_bar@tlsgd(%rip), %rdi
	.byte 0x66, 0x66, 0x48
	call __tls_get_addr@plt
	mov (%rax), %eax
	cmp $0x34, %eax
	je 1f
	mov $10, %eax
	jmp 2f
1:

# Test Global Dynamic TLS Model (large code model)
	lea tls_foo@tlsgd(%rip), %rdi
	movabs $__tls_get_addr@pltoff, %rax
	add %rbx, %rax
	call *%rax
	mov (%rax), %eax
	cmp $0x12, %eax
	je 1f
	mov $11, %eax
	jmp 2f
1:

	lea tls_bar@tlsgd(%rip), %rdi
	movabs $__tls_get_addr@pltoff, %rax
	add %rbx, %rax
	call *%rax
	mov (%rax), %eax
	cmp $0x34, %eax
	je 1f
	mov $12, %eax
	jmp 2f
1:

	xor %eax, %eax

2:
	pop %rbx
	ret


	.section .tdata, "awT", @progbits

	.global tls_foo
	.type tls_foo, @object
	.size tls_foo, 4
	.align 4
tls_foo:
	.long 0x12

	.global tls_bar
	.type tls_bar, @object
	.size tls_bar, 4
	.align 4
tls_bar:
	.long 0x34
