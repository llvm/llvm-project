# RUN: yaml2obj %s -o %t.o
# RUN: llvm-dwarfdump --debug-info=0x00000020 -p -parent-recurse-depth 0 %t.o | FileCheck %s --check-prefixes=COMMON,ALL
# RUN: llvm-dwarfdump --debug-info=0x00000020 -p -parent-recurse-depth 1 %t.o | FileCheck %s --check-prefixes=COMMON,ONE
# RUN: llvm-dwarfdump --debug-info=0x00000020 -p -parent-recurse-depth 2 %t.o | FileCheck %s --check-prefixes=COMMON,TWO
# RUN: llvm-dwarfdump --debug-info=0x00000020 -p -parent-recurse-depth 3 %t.o | FileCheck %s --check-prefixes=COMMON,ALL

# COMMON: .o: file format

# ALL: by_hand
# ALL: main
# ALL: test
# ALL: int

# ONE-NOT: by_hand
# ONE-NOT: main
# ONE: test
# ONE: int

# TWO-NOT: by_hand
# TWO: main
# TWO: test
# TWO: int

--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
DWARF:
  debug_abbrev:
    - Table:
      - Tag:      DW_TAG_compile_unit
        Children: DW_CHILDREN_yes
        Attributes:
          - Attribute: DW_AT_producer
            Form:      DW_FORM_string
      - Tag:      DW_TAG_subprogram
        Children: DW_CHILDREN_yes
        Attributes:
          - Attribute: DW_AT_name
            Form:      DW_FORM_string
      - Tag:      DW_TAG_namespace
        Children: DW_CHILDREN_yes
        Attributes:
          - Attribute: DW_AT_name
            Form:      DW_FORM_string
      - Tag:      DW_TAG_base_type
        Children: DW_CHILDREN_no
        Attributes:
          - Attribute: DW_AT_name
            Form:      DW_FORM_string
  debug_info:
    - Version: 4
      Entries:
        - AbbrCode: 1
          Values:
            - CStr: by_hand
        - AbbrCode: 2
          Values:
            - CStr: main
        - AbbrCode: 3
          Values:
            - CStr: test
        - AbbrCode: 4
          Values:
            - CStr: int
