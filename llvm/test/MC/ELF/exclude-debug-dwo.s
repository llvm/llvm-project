# RUN: llvm-mc -triple x86_64-pc-linux-gnu %s -filetype=obj -o %t.o
# RUN: llvm-readelf -S %t.o | FileCheck %s

## Check we add SHF_EXCLUDE flag for .dwo debug sections.
# CHECK: .debug_info.dwo        {{.*}} E
# CHECK: .debug_types.dwo       {{.*}} E
# CHECK: .debug_abbrev.dwo      {{.*}} E
# CHECK: .debug_str.dwo         {{.*}} MSE
# CHECK: .debug_line.dwo        {{.*}} E
# CHECK: .debug_loc.dwo         {{.*}} E
# CHECK: .debug_str_offsets.dwo {{.*}} E

.section .debug_info.dwo,"e"
nop

.section .debug_types.dwo,"e"
nop

.section .debug_abbrev.dwo,"e"
nop

.section .debug_str.dwo,"MSe",@progbits,1
nop

.section .debug_line.dwo,"e"
nop

.section .debug_loc.dwo,"e"
nop

.section .debug_str_offsets.dwo,"e"
nop
