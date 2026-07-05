## Test that llvm-dwarfdump --verify correctly handles DW_TAG_call_site
## nested inside a DW_TAG_lexical_block. Previously, the parent-walking
## loop in verifyDebugInfoCallSite() used Die.getParent() instead of
## Curr.getParent(), causing an infinite loop when the call_site's
## immediate parent was not a subprogram (e.g., a lexical_block).

# RUN: yaml2obj %s -o - | llvm-dwarfdump --verify - 2>&1 | FileCheck %s

# CHECK: No errors.

--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
DWARF:
  debug_str:
    - "callee"
    - "caller"

  debug_abbrev:
    - Table:
      - Code:            0x00000001
        Tag:             DW_TAG_compile_unit
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
      - Code:            0x00000002
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_data4
      - Code:            0x00000003
        Tag:             DW_TAG_subprogram
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_name
            Form:            DW_FORM_strp
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_data4
          - Attribute:       DW_AT_call_all_calls
            Form:            DW_FORM_flag_present
      - Code:            0x00000004
        Tag:             DW_TAG_lexical_block
        Children:        DW_CHILDREN_yes
        Attributes:
          - Attribute:       DW_AT_low_pc
            Form:            DW_FORM_addr
          - Attribute:       DW_AT_high_pc
            Form:            DW_FORM_data4
      - Code:            0x00000005
        Tag:             DW_TAG_call_site
        Children:        DW_CHILDREN_no
        Attributes:
          - Attribute:       DW_AT_call_origin
            Form:            DW_FORM_ref4

  debug_info:
    - Version:         5
      UnitType:        DW_UT_compile
      AbbrOffset:      0x00000000
      AddrSize:        8
      Entries:
        # DW_TAG_compile_unit
        - AbbrCode:    0x00000001
          Values:
            - Value:   0x0000000000000000   # DW_AT_low_pc
        # DW_TAG_subprogram (callee) - offset will be 0x0c + some bytes
        - AbbrCode:    0x00000002
          Values:
            - Value:   0x0000000000000000   # DW_AT_name -> "callee"
            - Value:   0x0000000000003000   # DW_AT_low_pc
            - Value:   0x0000000000000010   # DW_AT_high_pc
        # DW_TAG_subprogram (caller, with DW_AT_call_all_calls)
        - AbbrCode:    0x00000003
          Values:
            - Value:   0x0000000000000007   # DW_AT_name -> "caller"
            - Value:   0x0000000000001000   # DW_AT_low_pc
            - Value:   0x0000000000000100   # DW_AT_high_pc
        # DW_TAG_lexical_block (child of caller)
        - AbbrCode:    0x00000004
          Values:
            - Value:   0x0000000000001000   # DW_AT_low_pc
            - Value:   0x0000000000000080   # DW_AT_high_pc
        # DW_TAG_call_site (child of lexical_block)
        - AbbrCode:    0x00000005
          Values:
            - Value:   0x0000000000000015   # DW_AT_call_origin -> callee subprogram
        - AbbrCode:    0x00000000           # end lexical_block
          Values:      []
        - AbbrCode:    0x00000000           # end caller subprogram
          Values:      []
        - AbbrCode:    0x00000000           # end compile_unit
          Values:      []
...
