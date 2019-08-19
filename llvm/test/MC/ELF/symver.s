# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o
# RUN: llvm-readelf -s %t.o | FileCheck --check-prefix=SYM %s
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=REL %s

.globl global1_impl, global2_impl, global3_impl
local1_impl:
local2_impl:
local3_impl:
global1_impl:
global2_impl:
global3_impl:

# SYM:      0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
# SYM-NEXT: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT    2 local1@zed
# SYM-NEXT: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT    2 local2@@zed
# SYM-NEXT: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT    2 local3@@zed
# SYM-NEXT: 0000000000000000     0 SECTION LOCAL  DEFAULT    2
.symver local1_impl, local1@zed
.symver local2_impl, local2@@zed
.symver local3_impl, local3@@@zed

# SYM-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT    2 global1@zed
# SYM-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT    2 global2@@zed
# SYM-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT    2 global3@@zed
.symver global1_impl, global1@zed
.symver global2_impl, global2@@zed
.symver global3_impl, global3@@@zed

## undef3_impl@@@zed emits a non-default versioned undef3_impl@zed.
# SYM-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND undef1@zed
# SYM-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND undef3@zed
.symver undef1_impl, undef1@zed
.symver undef3_impl, undef3@@@zed

## GNU as emits {local,global}{1,2}_impl (.symver acted as a copy) but not
## {local,global}3_impl or undef{1,2,3}_impl (.symver acted as a rename).
## We consistently treat .symver as a rename and suppress the original symbols.
## This is advantageous because the original symbols are usually undesired
## and can easily cause issues like binutils PR/18703.
## If they want to retain the original symbol, 
# SYM-NOT: {{.}}

# REL:      Relocations [
# REL-NEXT:   Section {{.*}} .rela.text {
# REL-NEXT:     0x0 R_X86_64_32 .text 0x0
# REL-NEXT:     0x4 R_X86_64_32 .text 0x0
# REL-NEXT:     0x8 R_X86_64_32 .text 0x0
# REL-NEXT:     0xC R_X86_64_32 global1@zed 0x0
# REL-NEXT:     0x10 R_X86_64_32 global2@@zed 0x0
# REL-NEXT:     0x14 R_X86_64_32 global3@@zed 0x0
# REL-NEXT:     0x18 R_X86_64_32 undef1@zed 0x0
# REL-NEXT:     0x1C R_X86_64_32 undef3@zed 0x0
# REL-NEXT:   }
# REL-NEXT: ]
.long local1_impl
.long local2_impl
.long local3_impl
.long global1_impl
.long global2_impl
.long global3_impl
.long undef1_impl
.long undef3_impl
