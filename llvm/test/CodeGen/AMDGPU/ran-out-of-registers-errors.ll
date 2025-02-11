; RUN: not llc -mtriple=amdgcn-amd-amdhsa -stress-regalloc=1 -vgpr-regalloc=greedy -filetype=null %s 2>&1 | FileCheck -check-prefixes=CHECK,GREEDY -implicit-check-not=error %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -stress-regalloc=1 -vgpr-regalloc=basic -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error -check-prefixes=CHECK,BASIC %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -stress-regalloc=1 -vgpr-regalloc=fast -filetype=null %s 2>&1 | FileCheck -implicit-check-not=error -check-prefixes=CHECK,FAST %s
; RUN: opt -passes=debugify -o %t.bc %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -stress-regalloc=1 -vgpr-regalloc=greedy -filetype=null %t.bc 2>&1 | FileCheck -implicit-check-not=error -check-prefixes=DBGINFO-CHECK,DBGINFO-GREEDY %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -stress-regalloc=1 -vgpr-regalloc=basic -filetype=null %t.bc 2>&1 | FileCheck -implicit-check-not=error -check-prefixes=DBGINFO-CHECK,DBGINFO-BASIC %s

; FIXME: Asserts when using -O2 + -vgpr-regalloc=fast
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -stress-regalloc=1 -O0 -filetype=null %t.bc 2>&1 | FileCheck -implicit-check-not=error -check-prefixes=DBGINFO-CHECK,DBGINFO-FAST %s

; TODO: Should we fix emitting multiple errors sometimes in basic and fast?


; CHECK: error: <unknown>:0:0: ran out of registers during register allocation in function 'ran_out_of_registers_general'

; DBGINFO-GREEDY: error: {{.*}}:3:1: ran out of registers during register allocation in function 'ran_out_of_registers_general'

; DBGINFO-BASIC: error: {{.*}}:1:1: ran out of registers during register allocation in function 'ran_out_of_registers_general'

; DBGINFO-FAST: error: {{.*}}:3:1: ran out of registers during register allocation in function 'ran_out_of_registers_general'
define i32 @ran_out_of_registers_general(ptr addrspace(1) %ptr) #0 {
  %ld0 = load volatile i32, ptr addrspace(1) %ptr
  %ld1 = load volatile i32, ptr addrspace(1) %ptr
  %add = add i32 %ld0, %ld1
  ret i32 %add
}

; CHECK: error: inline assembly requires more registers than available at line 23
; DBGINFO-CHECK: error: inline assembly requires more registers than available at line 23
define void @ran_out_of_registers_asm_def() #0 {
  %asm = call { i32, i32 } asm sideeffect "; def $0 $1", "=v,=v"(), !srcloc !0
  ret void
}

; CHECK: error: inline assembly requires more registers than available at line 23
; DBGINFO-CHECK: error: inline assembly requires more registers than available at line 23
define void @ran_out_of_registers_asm_use() #0 {
  call void asm sideeffect "; def $0 $1", "v,v"(i32 0, i32 1), !srcloc !0
  ret void
}

; Test error in anonymous function.

; GREEDY: error: inline assembly requires more registers than available at line 23
; BASIC: error: inline assembly requires more registers than available at line 23

; FAST: error: <unknown>:0:0: ran out of registers during register allocation in function '@0'

; DBGINFO-GREEDY: error: inline assembly requires more registers than available at line 23
; DBGINFO-BASIC: error: inline assembly requires more registers than available at line 23

; DBGINFO-FAST: error: {{.*}}:12:1: ran out of registers during register allocation in function '@0'
define i32 @0(ptr addrspace(1) %ptr) #0 {
  %asm = call { i32, i32 } asm sideeffect "; def $0 $1 use $2", "=v,=v,v"(ptr addrspace(1) %ptr), !srcloc !0
  %elt0 = extractvalue { i32, i32 } %asm, 0
  %elt1 = extractvalue { i32, i32 } %asm, 1
  %add = add i32 %elt0, %elt1
  ret i32 %add
}

attributes #0 = { "target-cpu"="gfx908" }

!0 = !{i32 23}
