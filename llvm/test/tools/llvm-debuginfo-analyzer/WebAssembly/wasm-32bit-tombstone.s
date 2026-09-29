# REQUIRES: webassembly-registered-target

# Test that DWARF tombstones are correctly detected/respected in wasm
# 32 bit object files.

# The test case was produced by the following steps:
#
# // test-clang.cpp
# void foo() {
# }
#
# 1) clang --target=wasm32 -S -g test-clang.cpp
#                             -o Inputs/wasm-32bit-tombstone.s
#
# 2) Creating a single function, tombstoning it in the assembly, by
#    manually changing the DW_AT_low_pc for the DW_TAG_subprogram:
#    .Lfunc_begin0 to 0xffffffff to mark the function as dead code:
#
#	   .int8	2                          # Abbrev [2] 0x26:0x1b DW_TAG_subprogram
#	   .int32	.Lfunc_begin0              # DW_AT_low_pc  <---------
#	   .int32	.Lfunc_end0-.Lfunc_begin0  # DW_AT_high_pc

#	   .int8	2                          # Abbrev [2] 0x26:0x1b DW_TAG_subprogram
#	   .int32	0xffffffff                 # DW_AT_low_pc  <---------
#	   .int32	.Lfunc_end0-.Lfunc_begin0  # DW_AT_high_pc

# RUN: llvm-mc -arch=wasm32 -filetype=obj       \
# RUN:         %p/wasm-32bit-tombstone.s        \
# RUN:         -o %t.wasm-32bit-tombstone.wasm

# RUN: llvm-debuginfo-analyzer --select-elements=Discarded         \
# RUN:                         --print=elements                    \
# RUN:                         %t.wasm-32bit-tombstone.wasm 2>&1 | \
# RUN: FileCheck --strict-whitespace -check-prefix=ONE %s

# ONE: Logical View:
# ONE-NEXT:           {File} '{{.*}}wasm-32bit-tombstone.wasm'
# ONE-EMPTY:
# ONE-NEXT:           {CompileUnit} 'test-clang.cpp'
# ONE-NEXT:           {Function} not_inlined 'foo' -> 'void'

# RUN: llvm-dwarfdump --debug-info %t.wasm-32bit-tombstone.wasm | \
# RUN: FileCheck %s --check-prefix=TWO

# TWO:      DW_TAG_subprogram
# TWO-NEXT:   DW_AT_low_pc	(dead code)
# TWO-NEXT:   DW_AT_high_pc
# TWO-NEXT:   DW_AT_name	("foo")

	.text
	.file	"test-clang.cpp"
	.functype	_Z3foov () -> ()
	.section	.text._Z3foov,"",@
_Z3foov:                                # @_Z3foov
.Lfunc_begin0:
	.functype	_Z3foov () -> ()
	return
	end_function
.Lfunc_end0:
                                        # -- End function
	.section	.debug_abbrev,"",@
	.int8	1                               # Abbreviation Code
	.int8	17                              # DW_TAG_compile_unit
	.int8	1                               # DW_CHILDREN_yes
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	2                               # Abbreviation Code
	.int8	46                              # DW_TAG_subprogram
	.int8	0                               # DW_CHILDREN_no
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	0                               # EOM(3)
	.section	.debug_info,"",@
.Lcu_begin0:
	.int32	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.int16	4                               # DWARF version number
	.int32	.debug_abbrev0                  # Offset Into Abbrev. Section
	.int8	4                               # Address Size (in bytes)
	.int8	1                               # Abbrev [1] 0xb:0x37 DW_TAG_compile_unit
	.int32	.Linfo_string1                  # DW_AT_name
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	2                               # Abbrev [2] 0x26:0x1b DW_TAG_subprogram
	.int32	0xffffffff                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int32	.Linfo_string4                  # DW_AT_name
	.int8	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"S",@
.Linfo_string1:
	.asciz	"test-clang.cpp"                # string offset=176
.Linfo_string4:
	.asciz	"foo"                           # string offset=241
	.ident	"clang version 19.0.0"
