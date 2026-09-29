# RUN: llvm-mc -triple x86_64-unknown-darwin %s | FileCheck %s

# CHECK: __DATA,__thread_init,thread_local_init_function_pointers
# CHECK: .quad 0

.thread_init_func
	.quad 0
