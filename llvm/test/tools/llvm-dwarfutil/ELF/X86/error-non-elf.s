## Reject non-ELF objects before linking, copying or splitting debug info.
## Non-empty sections used to reach an unchecked ELFSectionRef cast.

# RUN: llvm-mc -triple=x86_64-apple-darwin -filetype=obj %s -o %t.macho
# RUN: not llvm-dwarfutil %t.macho %t.out 2>&1 | FileCheck %s
# RUN: not llvm-dwarfutil --linker=parallel %t.macho %t.out 2>&1 | FileCheck %s
# RUN: not llvm-dwarfutil --no-garbage-collection --no-odr --build-accelerator=none %t.macho %t.out 2>&1 | FileCheck %s
# RUN: not llvm-dwarfutil --no-garbage-collection --no-odr --build-accelerator=DWARF %t.macho %t.out 2>&1 | FileCheck %s
# RUN: not llvm-dwarfutil --no-garbage-collection --no-odr --build-accelerator=none --separate-debug-file %t.macho %t.out 2>&1 | FileCheck %s
# RUN: llvm-mc -triple=x86_64-pc-windows-msvc -filetype=obj %s -o %t.coff
# RUN: not llvm-dwarfutil %t.coff %t.out 2>&1 | FileCheck %s
# RUN: %python -c "open(r'%t.wasm', 'wb').write(bytes.fromhex('0061736d01000000'))"
# RUN: not llvm-dwarfutil %t.wasm %t.out 2>&1 | FileCheck %s

# CHECK: error: '{{.*}}': unsupported input file

.data
.long 42
