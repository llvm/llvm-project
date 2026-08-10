void func() {

#ifdef __powerpc64__
  __asm__ (
    "mflr 0;"
    "std 0,16(1);"
    "addi 1,1,-24;"
    "mr 31,1;"
    ".cfi_def_cfa_offset 24;"
    "addi 0,0,0;"
    "addi 1,1,24;"
    "ld 0,16(1);"
    ".cfi_def_cfa_offset 0;"
  );
#elif !defined __mips__
	__asm__ (
		"pushq $0x10;"
		".cfi_def_cfa_offset 16;"
		"jmp label;"
		"movq $0x48, %rax;"
"label: subq $0x38, %rax;"
		"movq $0x48, %rcx;"
		"movq $0x48, %rdx;"
		"movq $0x48, %rax;"
		"popq %rax;"
	);
#elif __mips64
   __asm__ (
    "daddiu $sp,$sp,-16;"
    ".cfi_def_cfa_offset 16;"
    "sd $ra,8($sp);"
    ".cfi_offset 31, -8;"
    "daddiu $ra,$zero,0;"
    "ld $ra,8($sp);"
    "daddiu $sp, $sp,16;"
    ".cfi_restore 31;"
    ".cfi_def_cfa_offset 0;"
   );
#else
   // For MIPS32
   __asm__ (
    "addiu $sp,$sp,-8;"
    ".cfi_def_cfa_offset 8;"
    "sw $ra,4($sp);"
    ".cfi_offset 31, -4;"
    "addiu $ra,$zero,0;"
    "lw $ra,4($sp);"
    "addiu $sp,$sp,8;"
    ".cfi_restore 31;"
    ".cfi_def_cfa_offset 0;"
   );
#endif
}

int main(int argc, char const *argv[])
{
	func();
}
