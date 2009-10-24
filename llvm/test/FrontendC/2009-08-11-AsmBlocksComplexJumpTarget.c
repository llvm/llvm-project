// RUN: %llvmgcc %s -fasm-blocks -S -o - | grep {\\\*1192}
// Complicated expression as jump target
// XFAIL: *
// XTARGET: x86,i386,i686

asm void Method3()
{
    mov   eax,[esp+4]           
    jmp   [eax+(299-1)*4]       
}
