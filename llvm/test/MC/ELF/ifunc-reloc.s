# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck %s
.global alias
.type alias, @function
alias:
  callq global

.global global
.type global, @gnu_indirect_function
.set global, alias

.type local, @gnu_indirect_function
.set local, alias

## Check we don't create the relocations with a section symbol for both global
## They may result in IRELATIVE relocs that the dynamic loader will use to
## resolve addresses at startup time.
.data
.quad global
.quad local

# CHECK:      Relocations [
# CHECK-NEXT:   Section {{.*}} .rela.text {
# CHECK-NEXT:     0x1 R_X86_64_PLT32 global 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {{.*}} .rela.data {
# CHECK-NEXT:     0x0 R_X86_64_64 global 0x0
# CHECK-NEXT:     0x8 R_X86_64_64 local 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
