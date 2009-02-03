; RUN: llvm-as < %s | llc -march=x86
; Test 'ri' constraint.

define void @run_init_process() {
          %tmp = call i32 asm sideeffect "push %ebx ; movl $2,%ebx ; int $$0x80 ; pop %ebx", "={ax},0,ri,{cx},{dx},~{dirflag},~{fpsr},~{flags},~{memory}"( i32 11, i32 0, i32 0, i32 0 )          
          unreachable
  }
