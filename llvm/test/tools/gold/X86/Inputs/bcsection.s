.global elf_func

elf_func:
   ret

.section .llvmbc
.incbin "bcsection.bc"
