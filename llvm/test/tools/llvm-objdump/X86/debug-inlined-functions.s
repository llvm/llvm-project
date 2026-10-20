## Generated with this compile command, with the source code in Inputs/debug-inlined-functions.cc:
## clang++ -g -c debug-inlined-functions.cc -O1 -S -o -

# RUN: llvm-mc -triple=x86_64 %s -filetype=obj -o %t.o

# RUN: llvm-objdump %t.o -d --debug-inlined-funcs=unicode | \
# RUN:     FileCheck %s --check-prefixes=UNICODE,UNICODE-MANGLED --strict-whitespace

# RUN: llvm-objdump %t.o -d -C --debug-inlined-funcs | \
# RUN:     FileCheck %s --check-prefixes=UNICODE,UNICODE-DEMANGLED --strict-whitespace

# RUN: llvm-objdump %t.o -d -C --debug-inlined-funcs=unicode | \
# RUN:     FileCheck %s --check-prefixes=UNICODE,UNICODE-DEMANGLED --strict-whitespace

# RUN: llvm-objdump %t.o -d -C --debug-inlined-funcs=unicode --debug-indent=30 | \
# RUN:     FileCheck %s --check-prefix=UNICODE-DEMANGLED-INDENT --strict-whitespace

# RUN: llvm-objdump %t.o -d -C --debug-inlined-funcs=ascii | \
# RUN:     FileCheck %s --check-prefix=ASCII-DEMANGLED --strict-whitespace

# RUN: llvm-objdump %t.o -d -C --debug-inlined-funcs=limits-only | \
# RUN:     FileCheck %s --check-prefix=LIMITS-ONLY-DEMANGLED

# RUN: llvm-objdump %t.o -d -C --debug-inlined-funcs=unicode --debug-vars=unicode | \
# RUN:     FileCheck %s --check-prefix=DEBUG-DEMANGLED-ALL --strict-whitespace

# UNICODE-MANGLED: 0000000000000000 <_Z3barii>:
# UNICODE-DEMANGLED: 0000000000000000 <bar(int, int)>:
# UNICODE-NEXT:        0: 8d 04 3e                     	leal	(%rsi,%rdi), %eax
# UNICODE-NEXT:        3: 0f af f7                     	imull	%edi, %esi
# UNICODE-NEXT:        6: 01 f0                        	addl	%esi, %eax
# UNICODE-NEXT:        8: c3                           	retq
# UNICODE-NEXT:        9: 0f 1f 80 00 00 00 00         	nopl	(%rax)
# UNICODE-EMPTY:
# UNICODE-MANGLED-NEXT: 0000000000000010 <_Z3fooii>:
# UNICODE-DEMANGLED-NEXT: 0000000000000010 <foo(int, int)>:
# UNICODE-MANGLED-NEXT:                                                                                     ┠─ _Z3barii = inlined into _Z3fooii
# UNICODE-DEMANGLED-NEXT:                                                                                   ┠─ bar(int, int) = inlined into foo(int, int)
# UNICODE-NEXT:      10: 8d 04 3e                     	leal	(%rsi,%rdi), %eax                           ┃
# UNICODE-NEXT:      13: 0f af f7                     	imull	%edi, %esi                                  ┃
# UNICODE-NEXT:      16: 01 f0                        	addl	%esi, %eax                                  ┻
# UNICODE-NEXT:      18: c3                           	retq

# UNICODE-DEMANGLED-INDENT: 0000000000000010 <foo(int, int)>:
# UNICODE-DEMANGLED-INDENT-NEXT:                                                                          ┠─ bar(int, int) = inlined into foo(int, int)
# UNICODE-DEMANGLED-INDENT-NEXT:       10: 8d 04 3e                     	leal	(%rsi,%rdi), %eax     ┃
# UNICODE-DEMANGLED-INDENT-NEXT:       13: 0f af f7                     	imull	%edi, %esi            ┃
# UNICODE-DEMANGLED-INDENT-NEXT:       16: 01 f0                        	addl	%esi, %eax            ┻
# UNICODE-DEMANGLED-INDENT-NEXT:       18: c3                           	retq

# ASCII-DEMANGLED: 0000000000000010 <foo(int, int)>:
# ASCII-DEMANGLED-NEXT:                                                                                                 |- bar(int, int) = inlined into foo(int, int)
# ASCII-DEMANGLED-NEXT:        10: 8d 04 3e                     	leal	(%rsi,%rdi), %eax                           |
# ASCII-DEMANGLED-NEXT:        13: 0f af f7                     	imull	%edi, %esi                                  |
# ASCII-DEMANGLED-NEXT:        16: 01 f0                        	addl	%esi, %eax                                  v
# ASCII-DEMANGLED-NEXT:        18: c3                           	retq

# LIMITS-ONLY-DEMANGLED: 0000000000000010 <foo(int, int)>:
# LIMITS-ONLY-DEMANGLED-NEXT: debug-inlined-functions.cc:8:16: bar(int, int) inlined into foo(int, int)
# LIMITS-ONLY-DEMANGLED-NEXT: 10: 8d 04 3e                     leal    (%rsi,%rdi), %eax
# LIMITS-ONLY-DEMANGLED-NEXT: 13: 0f af f7                     imull   %edi, %esi
# LIMITS-ONLY-DEMANGLED-NEXT: 16: 01 f0                        addl    %esi, %eax
# LIMITS-ONLY-DEMANGLED-NEXT: debug-inlined-functions.cc:8:16: end of bar(int, int) inlined into foo(int, int)
# LIMITS-ONLY-DEMANGLED-NEXT: 18: c3                           retq

# DEBUG-DEMANGLED-ALL: 0000000000000010 <foo(int, int)>:
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┠─ a = RDI
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┃ ┠─ b = RSI
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┃ ┃ ┠─ bar(int, int) = inlined into foo(int, int)
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┃ ┃ ┃ ┠─ x = RDI
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┃ ┃ ┃ ┃ ┠─ y = RSI
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┃ ┃ ┃ ┃ ┃ ┌─ sum = RAX
# DEBUG-DEMANGLED-ALL-NEXT:  10: 8d 04 3e                     	leal	(%rsi,%rdi), %eax                           ┃ ┃ ┃ ┃ ┃ ╈
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┃ ┃ ┃ ┃ ┃ ┃ ┌─ b = entry(RSI)
# DEBUG-DEMANGLED-ALL-NEXT:                                                                                           ┃ ┃ ┃ ┃ ┃ ┃ │ ┌─ mul = RSI
# DEBUG-DEMANGLED-ALL-NEXT:  13: 0f af f7                     	imull	%edi, %esi                                  ┃ ┻ ┃ ┃ ┻ ┃ ╈ ╈
# DEBUG-DEMANGLED-ALL-NEXT:  																							┃ ┌─ result = RAX
# DEBUG-DEMANGLED-ALL-NEXT:  16: 01 f0                        	addl	%esi, %eax                                  ┃ ╈ ┻ ┻   ┻ ┃ ┃
# DEBUG-DEMANGLED-ALL-NEXT:  18: c3                           	retq                                                ┻ ┻         ┻ ┻

	.file	"debug-inlined-functions.cc"
	.text
	.globl	_Z3barii                        # -- Begin function _Z3barii
	.p2align	4
	.type	_Z3barii,@function
_Z3barii:                               # @_Z3barii
.Lfunc_begin0:
	.file	0 "debug-inlined-functions.cc" md5 0xf07b869ec4d0996589aa6856ae4e6c83
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: bar:x <- $edi
	#DEBUG_VALUE: bar:y <- $esi
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	.loc	0 2 15 prologue_end             # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:2:15
	leal	(%rsi,%rdi), %eax
.Ltmp0:
	#DEBUG_VALUE: bar:sum <- $eax
	.loc	0 3 15                          # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:3:15
	imull	%edi, %esi
.Ltmp1:
	#DEBUG_VALUE: bar:y <- [DW_OP_LLVM_entry_value 1] $esi
	#DEBUG_VALUE: bar:mul <- $esi
	.loc	0 4 14                          # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:4:14
	addl	%esi, %eax
.Ltmp2:
	.loc	0 4 3 is_stmt 0                 # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:4:3
	retq
.Ltmp3:
.Lfunc_end0:
	.size	_Z3barii, .Lfunc_end0-_Z3barii
	.cfi_endproc
                                        # -- End function
	.globl	_Z3fooii                        # -- Begin function _Z3fooii
	.p2align	4
	.type	_Z3fooii,@function
_Z3fooii:                               # @_Z3fooii
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: foo:a <- $edi
	#DEBUG_VALUE: foo:b <- $esi
	#DEBUG_VALUE: bar:x <- $edi
	#DEBUG_VALUE: bar:y <- $esi
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	.loc	0 2 15 prologue_end is_stmt 1   # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:2:15 @[ llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:8:16 ]
	leal	(%rsi,%rdi), %eax
.Ltmp4:
	#DEBUG_VALUE: bar:sum <- $eax
	.loc	0 3 15                          # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:3:15 @[ llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:8:16 ]
	imull	%edi, %esi
.Ltmp5:
	#DEBUG_VALUE: foo:b <- [DW_OP_LLVM_entry_value 1] $esi
	#DEBUG_VALUE: bar:mul <- $esi
	.loc	0 4 14                          # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:4:14 @[ llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:8:16 ]
	addl	%esi, %eax
.Ltmp6:
	#DEBUG_VALUE: foo:result <- $eax
	.loc	0 9 3                           # llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc:9:3
	retq
.Ltmp7:
.Lfunc_end1:
	.size	_Z3fooii, .Lfunc_end1-_Z3fooii
	.cfi_endproc
                                        # -- End function
	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	8                               # Offset entry count
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
	.long	.Ldebug_loc1-.Lloclists_table_base0
	.long	.Ldebug_loc2-.Lloclists_table_base0
	.long	.Ldebug_loc3-.Lloclists_table_base0
	.long	.Ldebug_loc4-.Lloclists_table_base0
	.long	.Ldebug_loc5-.Lloclists_table_base0
	.long	.Ldebug_loc6-.Lloclists_table_base0
	.long	.Ldebug_loc7-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    #   starting offset
	.uleb128 .Ltmp1-.Lfunc_begin0           #   ending offset
	.byte	1                               # Loc expr size
	.byte	84                              # super-register DW_OP_reg4
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp1-.Lfunc_begin0           #   starting offset
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   ending offset
	.byte	4                               # Loc expr size
	.byte	163                             # DW_OP_entry_value
	.byte	1                               # 1
	.byte	84                              # super-register DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc1:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           #   starting offset
	.uleb128 .Ltmp2-.Lfunc_begin0           #   ending offset
	.byte	1                               # Loc expr size
	.byte	80                              # super-register DW_OP_reg0
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc2:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp1-.Lfunc_begin0           #   starting offset
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   ending offset
	.byte	1                               # Loc expr size
	.byte	84                              # super-register DW_OP_reg4
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc3:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Lfunc_begin1-.Lfunc_begin0    #   starting offset
	.uleb128 .Ltmp5-.Lfunc_begin0           #   ending offset
	.byte	1                               # Loc expr size
	.byte	84                              # super-register DW_OP_reg4
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp5-.Lfunc_begin0           #   starting offset
	.uleb128 .Lfunc_end1-.Lfunc_begin0      #   ending offset
	.byte	4                               # Loc expr size
	.byte	163                             # DW_OP_entry_value
	.byte	1                               # 1
	.byte	84                              # super-register DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc4:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Lfunc_begin1-.Lfunc_begin0    #   starting offset
	.uleb128 .Ltmp5-.Lfunc_begin0           #   ending offset
	.byte	1                               # Loc expr size
	.byte	84                              # super-register DW_OP_reg4
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc5:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp4-.Lfunc_begin0           #   starting offset
	.uleb128 .Ltmp6-.Lfunc_begin0           #   ending offset
	.byte	1                               # Loc expr size
	.byte	80                              # super-register DW_OP_reg0
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc6:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp5-.Lfunc_begin0           #   starting offset
	.uleb128 .Lfunc_end1-.Lfunc_begin0      #   ending offset
	.byte	1                               # Loc expr size
	.byte	84                              # super-register DW_OP_reg4
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc7:
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp6-.Lfunc_begin0           #   starting offset
	.uleb128 .Lfunc_end1-.Lfunc_begin0      #   ending offset
	.byte	1                               # Loc expr size
	.byte	80                              # super-register DW_OP_reg0
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.ascii	"\214\001"                      # DW_AT_loclists_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0xc4 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lloclists_table_base0          # DW_AT_loclists_base
	.byte	2                               # Abbrev [2] 0x27:0x26 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.long	77                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x33:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	86                              # DW_AT_abstract_origin
	.byte	4                               # Abbrev [4] 0x3a:0x6 DW_TAG_formal_parameter
	.byte	0                               # DW_AT_location
	.long	94                              # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0x40:0x6 DW_TAG_variable
	.byte	1                               # DW_AT_location
	.long	102                             # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0x46:0x6 DW_TAG_variable
	.byte	2                               # DW_AT_location
	.long	110                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x4d:0x2a DW_TAG_subprogram
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
                                        # DW_AT_external
                                        # DW_AT_inline
	.byte	7                               # Abbrev [7] 0x56:0x8 DW_TAG_formal_parameter
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x5e:0x8 DW_TAG_formal_parameter
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x66:0x8 DW_TAG_variable
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x6e:0x8 DW_TAG_variable
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x77:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	10                              # Abbrev [10] 0x7b:0x54 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	10                              # DW_AT_linkage_name
	.byte	11                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0x8b:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	12                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x95:0x9 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_location
	.byte	13                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x9e:0x9 DW_TAG_variable
	.byte	7                               # DW_AT_location
	.byte	14                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0xa7:0x27 DW_TAG_inlined_subroutine
	.long	77                              # DW_AT_abstract_origin
	.byte	1                               # DW_AT_low_pc
	.long	.Ltmp6-.Lfunc_begin1            # DW_AT_high_pc
	.byte	0                               # DW_AT_call_file
	.byte	8                               # DW_AT_call_line
	.byte	16                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0xb4:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	86                              # DW_AT_abstract_origin
	.byte	4                               # Abbrev [4] 0xbb:0x6 DW_TAG_formal_parameter
	.byte	4                               # DW_AT_location
	.long	94                              # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0xc1:0x6 DW_TAG_variable
	.byte	5                               # DW_AT_location
	.long	102                             # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0xc7:0x6 DW_TAG_variable
	.byte	6                               # DW_AT_location
	.long	110                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	64                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 21.0.0git (git@github.com:llvm/llvm-project.git eed98e1493414ae9c30596b1eeb8f4a9b260e42)" # string offset=0
.Linfo_string1:
	.asciz	"llvm/test/tools/llvm-objdump/X86/Inputs/debug-inlined-functions.cc" # string offset=112
.Linfo_string2:
	.asciz	"llvm-project" # string offset=179
.Linfo_string3:
	.asciz	"_Z3barii"                      # string offset=229
.Linfo_string4:
	.asciz	"bar"                           # string offset=238
.Linfo_string5:
	.asciz	"int"                           # string offset=242
.Linfo_string6:
	.asciz	"x"                             # string offset=246
.Linfo_string7:
	.asciz	"y"                             # string offset=248
.Linfo_string8:
	.asciz	"sum"                           # string offset=250
.Linfo_string9:
	.asciz	"mul"                           # string offset=254
.Linfo_string10:
	.asciz	"_Z3fooii"                      # string offset=258
.Linfo_string11:
	.asciz	"foo"                           # string offset=267
.Linfo_string12:
	.asciz	"a"                             # string offset=271
.Linfo_string13:
	.asciz	"b"                             # string offset=273
.Linfo_string14:
	.asciz	"result"                        # string offset=275
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:
	.ident	"clang version 21.0.0git (git@github.com:llvm/llvm-project.git eed98e1493414ae9c30596b1eeb8f4a9b260e42a)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
