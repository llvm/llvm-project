# Test handling of forward-declared (DW_AT_declaration) structures. These types
# can be produced due to vtable-based type homing, or other -flimit-debug-info
# optimizations.

# RUN: split-file %s %t
# RUN: llvm-mc --triple x86_64-pc-linux %t/asm --filetype=obj -o %t.o
# RUN: %lldb -o "settings set interpreter.stop-command-source-on-error false" \
# RUN:   -s %t/commands -o exit %t.o 2>&1 | FileCheck %s

#--- commands
# Type A should be treated as a forward-declaration even though it has a child.
target var a
# CHECK-LABEL: target var a
# FIXME: This should also produce some kind of an error.
# CHECK: (A) a = <incomplete type "A">
expr a
# CHECK-LABEL: expr a
# CHECK: incomplete type 'A' where a complete type is required

# Parsing B::B1 should not crash even though B is incomplete. Note that in this
# case B must be forcefully completed.
target var b1
# CHECK-LABEL: target var b1
# CHECK: (B::B1) b1 = (ptr = 0x00000000baadf00d)
expr b1
# CHECK-LABEL: expr b1
# CHECK: (B::B1) $0 = (ptr = 0x00000000baadf00d)

target var c1
# CHECK-LABEL: target var c1
# CHECK: (C::C1) c1 = 424742

expr c1
# CHECK-LABEL: expr c1
# CHECK: (C::C1) $1 = 424742
#--- asm
        .text
_ZN1AC2Ev:
        retq
.LZN1AC2Ev_end:

        .data
        .p2align 4
a:
        .quad   _ZTV1A+16
        .quad   0xdeadbeef

b1:
        .quad   0xbaadf00d
c1:
        .long   42474247

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   5                               # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   52                              # DW_AT_artificial
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
        .byte   2                               # DW_TAG_class_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   7                               # Abbreviation Code
        .byte   13                              # DW_TAG_member
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   56                              # DW_AT_data_member_location
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   8                               # Abbreviation Code
        .byte   15                              # DW_TAG_pointer_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   9                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   10                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
        .byte   100                             # DW_AT_object_pointer
        .byte   19                              # DW_FORM_ref4
        .byte   71                              # DW_AT_specification
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   11                              # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   52                              # DW_AT_artificial
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   12                              # Abbreviation Code
        .byte   22                              # DW_TAG_typedef
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                               # DWARF version number
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .byte   1                               # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .quad   _ZN1AC2Ev                       # DW_AT_low_pc
        .long   .LZN1AC2Ev_end-_ZN1AC2Ev        # DW_AT_high_pc

# Case 1: The compiler has omitted the declaration of the type, but it still
# produces an entry for its implicit constructor instantiated in this compile
# unit.
# Roughly corresponds to this:
# struct A {
#   virtual ~A(); // not defined here
#   // implicit A() {}
# } a;
        .byte   2                               # Abbrev [2] DW_TAG_variable
        .asciz  "a"                             # DW_AT_name
        .long   .LA-.Lcu_begin0                 # DW_AT_type
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   a
.LA:
        .byte   3                               # Abbrev [3] DW_TAG_structure_type
        .asciz  "A"                             # DW_AT_name
                                                # DW_AT_declaration
.LA_A:
        .byte   4                               # Abbrev [4] DW_TAG_subprogram
        .asciz  "A"                             # DW_AT_name
                                                # DW_AT_declaration
        .byte   5                               # Abbrev [5] DW_TAG_formal_parameter
        .long   .LAptr-.Lcu_begin0              # DW_AT_type
                                                # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
.LAptr:
        .byte   8                               # Abbrev [8] DW_TAG_pointer_type
        .long   .LA-.Lcu_begin0                 # DW_AT_type
        .byte   10                              # Abbrev [10] DW_TAG_subprogram
        .quad   _ZN1AC2Ev                       # DW_AT_low_pc
        .long   .LZN1AC2Ev_end-_ZN1AC2Ev        # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   86
        .long   .Lthis-.Lcu_begin0              # DW_AT_object_pointer
        .long   .LA_A-.Lcu_begin0               # DW_AT_specification
.Lthis:
        .byte   11                              # Abbrev [11] DW_TAG_formal_parameter
        .byte   2                               # DW_AT_location
        .byte   145
        .byte   120
        .asciz  "this"                          # DW_AT_name
        .long   .LAptr-.Lcu_begin0              # DW_AT_type
                                                # DW_AT_artificial
        .byte   0                               # End Of Children Mark

# Case 2: A structure has been emitted as a declaration only, but it contains a
# nested class, which has a full definition present.
# Rougly corresponds to this:
# struct B {
#   virtual ~B(); // not defined here
#   class B1 {
#     A* ptr;
#   };
# };
# B::B1 b1;
# Note that it is important that the inner type is a class (not struct) as that
# triggers a clang assertion.
        .byte   3                               # Abbrev [3] DW_TAG_structure_type
        .asciz  "B"                             # DW_AT_name
                                                # DW_AT_declaration
.LB1:
        .byte   6                               # Abbrev [6] DW_TAG_class_type
        .asciz  "B1"                            # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   7                               # Abbrev [5] 0x58:0xc DW_TAG_member
        .asciz  "ptr"                           # DW_AT_name
        .long   .LAptr                          # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark

        .byte   2                               # Abbrev [2] DW_TAG_variable
        .asciz  "b1"                            # DW_AT_name
        .long   .LB1-.Lcu_begin0                # DW_AT_type
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   b1

# Case 3: A typedef in DW_AT_declaration struct.
# C++ equivalent:
# struct C {
#   virtual ~C(); // not defined here
#   typedef int C1;
# };
# C::C1 c1;
.Lint:
        .byte   9                               # Abbrev [9] DW_TAG_base_type
        .asciz  "int"                           # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   3                               # Abbrev [3] DW_TAG_structure_type
        .asciz  "C"                             # DW_AT_name
                                                # DW_AT_declaration
.LC1:
        .byte   12                              # Abbrev [12] DW_TAG_typedef
        .long   .Lint-.Lcu_begin0               # DW_AT_type
        .asciz  "C1"                            # DW_AT_name
        .byte   0                               # End Of Children Mark

        .byte   2                               # Abbrev [2] DW_TAG_variable
        .asciz  "c1"                            # DW_AT_name
        .long   .LC1-.Lcu_begin0                # DW_AT_type
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   c1

        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
