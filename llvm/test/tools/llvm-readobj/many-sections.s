## many-sections.elf-x86_64 is a file that was generated to simulate
## an object with more than ~65k sections. When an ELF object
## has SHN_LORESERVE (0xff00) or more sections, its e_shnum field
## should be zero and sh_size of the section header at index 0 is used
## to store the value. If the section name string table section index is
## greater than or equal to SHN_LORESERVE, then e_shstrndx field
## should have the value of SHN_XINDEX and sh_link of the section header
## at index 0 is used to store the value.
##
## many-sections.elf-x86_64 has few sections to save disk
## space, but its e_shnum, e_shstrndx, sh_size and sh_link fields are set
## according to the above description, so that we can test the dumper.

# RUN: llvm-readelf --file-headers -S %p/Inputs/many-sections.elf-x86_64 | \
# RUN:   FileCheck %s --check-prefix=GNU1
# GNU1: Number of section headers:         0 (5)
# GNU1: Section header string table index: 65535 (3)
# GNU1: There are 5 section headers, starting at offset 0xb8

# RUN: llvm-readobj --file-headers %p/Inputs/many-sections.elf-x86_64 | \
# RUN:   FileCheck %s --check-prefix=LLVM1
# LLVM1: SectionHeaderCount: 0 (5)
# LLVM1: StringTableSectionIndex: 65535 (3)

## many-sections-stripped.elf-x86_64 is many-sections.elf-x86_64 with
## e_shoff field set to zero, but not e_shstrndx, to show that
## this corrupt case is handled correctly.

# RUN: llvm-readelf --file-headers %p/Inputs/many-sections-stripped.elf-x86_64 | \
# RUN:   FileCheck %s --check-prefix=GNU2
# GNU2: Number of section headers:         0
# GNU2: Section header string table index: 65535 (corrupt: out of range)

# RUN: llvm-readobj --file-headers %p/Inputs/many-sections-stripped.elf-x86_64 | \
# RUN:   FileCheck %s --check-prefix=LLVM2
# LLVM2: SectionHeaderCount: 0
# LLVM2: StringTableSectionIndex: 65535 (corrupt: out of range)
