# REQUIRES: x86-registered-target

## Test that we report meaningful warnings when dumping
## broken Android's packed relocation sections.

# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/asm1.s -o %t1.o
# RUN: llvm-readobj --relocations %t1.o 2>&1 | FileCheck %s -DFILE=%t1.o --check-prefix=ERR-HEADER
# RUN: llvm-readelf --relocations %t1.o 2>&1 | FileCheck %s -DFILE=%t1.o --check-prefix=ERR-HEADER

#--- asm1.s
.section .rela.dyn, "a", @0x60000001
.ascii "APS9"

# ERR-HEADER: warning: '[[FILE]]': unable to read relocations from SHT_ANDROID_REL section with index 3: invalid packed relocation header

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/asm2.s -o %t2.o
# RUN: llvm-readobj --relocations %t2.o 2>&1 | FileCheck %s -DFILE=%t2.o --check-prefix=ERR-PAST-END
# RUN: llvm-readelf --relocations %t2.o 2>&1 | FileCheck %s -DFILE=%t2.o --check-prefix=ERR-PAST-END

#--- asm2.s
.section .rela.dyn, "a", @0x60000001
.ascii "APS2"

# ERR-PAST-END: warning: '[[FILE]]': unable to read relocations from SHT_ANDROID_REL section with index 3: unable to decode LEB128 at offset 0x{{([[:xdigit:]]{8})}}: malformed sleb128, extends past end

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/asm3.s -o %t3.o
# RUN: llvm-readobj --relocations %t3.o 2>&1 | FileCheck %s -DFILE=%t3.o --check-prefix=ERR-PAST-END
# RUN: llvm-readelf --relocations %t3.o 2>&1 | FileCheck %s -DFILE=%t3.o --check-prefix=ERR-PAST-END

#--- asm3.s
.section .rela.dyn, "a", @0x60000001
.ascii "APS2"
.sleb128 4 ## Number of relocations
.sleb128 0 ## Initial offset

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/asm4.s -o %t4.o
# RUN: llvm-readobj --relocations %t4.o 2>&1 | FileCheck %s -DFILE=%t4.o --check-prefix=ERR-PAST-END
# RUN: llvm-readelf --relocations %t4.o 2>&1 | FileCheck %s -DFILE=%t4.o --check-prefix=ERR-PAST-END

#--- asm4.s
.section .rela.dyn, "a", @0x60000001
.ascii "APS2"
.sleb128 4 ## Number of relocations
.sleb128 0 ## Initial offset

.sleb128 2 ## Number of relocations in group
.sleb128 2 ## RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG
.sleb128 8 ## offset delta

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/asm5.s -o %t5.o
# RUN: llvm-readobj --relocations %t5.o 2>&1 | FileCheck %s -DFILE=%t5.o --check-prefix=ERR-LARGE
# RUN: llvm-readelf --relocations %t5.o 2>&1 | FileCheck %s -DFILE=%t5.o --check-prefix=ERR-LARGE

# ERR-LARGE: warning: '[[FILE]]': unable to read relocations from SHT_ANDROID_REL section with index 3: relocation group unexpectedly large

#--- asm5.s
.section .rela.dyn, "a", @0x60000001
.ascii "APS2"
.sleb128 4 ## Number of relocations
.sleb128 0 ## Initial offset

.sleb128 5 ## Number of relocations in group
.sleb128 2 ## RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG
.sleb128 8 ## offset delta
