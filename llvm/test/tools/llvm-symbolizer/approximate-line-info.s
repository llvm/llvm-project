# REQUIRES: x86-registered-target

# RUN: llvm-mc -g -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-symbolizer --obj=%t.o 0xa | FileCheck --check-prefix=APPROX-NONE %s
# RUN: llvm-symbolizer --obj=%t.o --approximate-line-info=before 0xa | FileCheck --check-prefix=APPROX-BEFORE %s
# RUN: llvm-symbolizer --obj=%t.o --approximate-line-info=after 0xa | FileCheck --check-prefix=APPROX-AFTER %s
# RUN: llvm-symbolizer --obj=%t.o --approximate-line-info=before 0xa 0x10 | FileCheck --check-prefixes=APPROX-BEFORE,NO-APPROX %s
# RUN: llvm-symbolizer --obj=%t.o --approximate-line-info=before --verbose 0xa | FileCheck --check-prefix=APPROX-VERBOSE %s
# RUN: llvm-symbolizer --obj=%t.o --approximate-line-info=before --output-style=JSON 0xa | FileCheck --check-prefix=APPROX-JSON %s

# APPROX-NONE: main
# APPROX-NONE-NEXT: {{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}main.c:0:6
# APPROX-BEFORE: main
# APPROX-BEFORE-NEXT: {{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}main.c:4:6 (approximate)
# APPROX-AFTER: main
# APPROX-AFTER-NEXT: {{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}main.c:8:2 (approximate)
# NO-APPROX: main
# NO-APPROX-NEXT: {{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}main.c:8:2

#APPROX-VERBOSE: main
#APPROX-VERBOSE-NEXT: Filename: /tmp/test{{[/|\]}}main.c
#APPROX-VERBOSE-NEXT: Function start address: 0x0
#APPROX-VERBOSE-NEXT: Line: 4
#APPROX-VERBOSE-NEXT: Column: 6
#APPROX-VERBOSE-NEXT: Approximate: 1

#APPROX-JSON: [{"Address":"0xa","ModuleName":"{{.*}}{{[/|\]+}}test{{[/|\]+}}tools{{[/|\]+}}llvm-symbolizer{{[/|\]+}}Output{{[/|\]+}}approximate-line-info.s.tmp.o","Symbol":[{"Approximate":true,"Column":6,"Discriminator":0,"FileName":"{{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}main.c","FunctionName":"main","Line":4,"StartAddress":"0x0","StartFileName":"","StartLine":0}]}]

## Generated from C Code
##
## int foo = 0;
## int x=89;
## int main() {
## if(x)
##  return foo;
## else
##  return x;
## }
##
## clang -S -O3 -gline-tables-only --target=x86_64-pc-linux

	.text
	.file	"main.c"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	0 "/tmp/test" "main.c" md5 0x26c3fbaea8e6febaf09ef44d37ec5ecc
	.cfi_startproc
# %bb.0:                                # %entry
	.loc	0 4 6 prologue_end              # main.c:4:6
	movl	x(%rip), %eax
	testl	%eax, %eax
	je	.LBB0_2
# %bb.1:                                # %entry
	.loc	0 0 6 is_stmt 0                 # main.c:0:6
	movl	foo(%rip), %eax
.LBB0_2:                                # %entry
	.loc	0 8 2 is_stmt 1                 # main.c:8:2
	retq
.Ltmp0:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	foo,@object                     # @foo
	.bss
	.globl	foo
	.p2align	2, 0x0
foo:
	.long	0                               # 0x0
	.size	foo, 4

	.type	x,@object                       # @x
	.data
	.globl	x
	.p2align	2, 0x0
x:
	.long	89                              # 0x59
	.size	x, 4

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
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
	.byte	1                               # Abbrev [1] 0xc:0x17 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	29                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	16                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:ampandey-1995/llvm-project.git 6751baed8d1ee8c5fd12fe5a06aa67275fc1ebf6)" # string offset=0
.Linfo_string1:
	.asciz	"main.c"                        # string offset=113
.Linfo_string2:
	.asciz	"/tmp/test"       # string offset=120
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"clang version 19.0.0git (git@github.com:ampandey-1995/llvm-project.git 6751baed8d1ee8c5fd12fe5a06aa67275fc1ebf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
