# Check we are able to dump the dynamic section without a DT_NULL entry correctly.

# RUN: yaml2obj -docnum=1 %s -o %t.o
# RUN: llvm-readobj --dynamic-table %t.o | FileCheck %s --check-prefix=NONULL-LLVM
# RUN: llvm-readelf --dynamic-table %t.o | FileCheck %s --check-prefix=NONULL-GNU

# NONULL-LLVM:      DynamicSection [ (1 entries)
# NONULL-LLVM-NEXT:   Tag                Type   Name/Value
# NONULL-LLVM-NEXT:   0x0000000000000015 DEBUG  0x0
# NONULL-LLVM-NEXT: ]

# NONULL-GNU:      Dynamic section at offset {{.*}} contains 1 entries:
# NONULL-GNU-NEXT:   Tag                Type     Name/Value
# NONULL-GNU-NEXT:   0x0000000000000015 (DEBUG)  0x0

--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
Sections:
  - Name:            .dynamic
    Type:            SHT_DYNAMIC
    Address:         0x0000000000001010
    AddressAlign:    0x0000000000000010
    EntSize:         0x0000000000000010
    Entries:
      - Tag:             DT_DEBUG
        Value:           0x0000000000000000
ProgramHeaders:
  - Type: PT_LOAD
    VAddr: 0x1000
    Sections:
      - Section: .dynamic
  - Type: PT_DYNAMIC
    VAddr: 0x1010
    Sections:
      - Section: .dynamic

# Sometimes .dynamic section content length can be greater than the
# length of its entries. In this case, we should not try to dump anything
# past the DT_NULL entry, which works as a terminator.

# RUN: yaml2obj -docnum=2 %s -o %t.o
# RUN: llvm-readobj --dynamic-table %t.o | FileCheck %s --check-prefix=LONG-LLVM
# RUN: llvm-readelf --dynamic-table %t.o | FileCheck %s --check-prefix=LONG-GNU

# LONG-LLVM:      DynamicSection [ (2 entries)
# LONG-LLVM-NEXT:   Tag                Type                 Name/Value
# LONG-LLVM-NEXT:   0x0000000000000015 DEBUG                0x0
# LONG-LLVM-NEXT:   0x0000000000000000 NULL                 0x0
# LONG-LLVM-NEXT: ]

# LONG-GNU:      Dynamic section at offset {{.*}} contains 2 entries:
# LONG-GNU-NEXT:   Tag                Type                 Name/Value
# LONG-GNU-NEXT:   0x0000000000000015 (DEBUG)              0x0
# LONG-GNU-NEXT:   0x0000000000000000 (NULL)               0x0

--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
Sections:
  - Name:            .dynamic
    Type:            SHT_DYNAMIC
    Address:         0x0000000000001010
    AddressAlign:    0x0000000000000010
    EntSize:         0x0000000000000010
    Entries:
      - Tag:             DT_DEBUG
        Value:           0x0000000000000000
      - Tag:             DT_NULL
        Value:           0x0000000000000000
      - Tag:             DT_NULL
        Value:           0x0000000000000000
ProgramHeaders:
  - Type: PT_LOAD
    VAddr: 0x1000
    Sections:
      - Section: .dynamic
  - Type: PT_DYNAMIC
    VAddr: 0x1010
    Sections:
      - Section: .dynamic
