; RUN: not llc -march=amdgcn < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: error: couldn't allocate output register for constraint 'q'
define void @crash_use_invalid_output_constraint_block(ptr addrspace(1) %arg) {
bb:
  %v = call i32 asm sideeffect "", "=q"()
  br label %use

use:
  store i32 %v, ptr addrspace(1) %arg
  ret void
}

; ERR: error: unknown asm constraint 'q'
define void @invalid_input_constraint() {
  call void asm sideeffect "", "q"(i32 1)
  ret void
}

; ERR: error: unknown asm constraint 'q'
define void @invalid_input_constraint_multi() {
  call void asm sideeffect "", "q,q"(i32 1, i16 2)
  ret void
}

; ERR: error: unknown asm constraint 'q'
define void @invalid_input_constraint_multi_valid() {
  call void asm sideeffect "", "q,v"(i32 1, i64 2)
  ret void
}

; ERR: error: couldn't allocate output register for constraint 'q'
define void @crash_use_invalid_output_constraint_block_multi(ptr addrspace(1) %arg) {
bb:
  %v = call { i32, i32 } asm sideeffect "", "=q,=q"()
  br label %use

use:
  store { i32, i32 } %v, ptr addrspace(1) %arg
  ret void
}

; ERR: error: couldn't allocate output register for constraint 'q'
define void @crash_use_invalid_output_constraint_block_multi_valid(ptr addrspace(1) %arg) {
bb:
  %v = call { i32, i32 } asm sideeffect "", "=q,=v"()
  br label %use

use:
  store { i32, i32 } %v, ptr addrspace(1) %arg
  ret void
}
