# This test generates a DW_TAG_structure_type with a linkage name. This linkage
# name will not be part of the accelerator table and the verifier should not
# complain about this.
#
# DW_TAG_structure_type
#   DW_AT_name    ("C")
#   DW_AT_linkage_name    ("$S4main1CCD")
#
# RUN: llvm-mc %s -filetype obj -triple x86_64-unknown-linux-gnu -o %t.o
# RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s
# RUN: llvm-dwarfdump -debug-names %t.o | FileCheck %s --check-prefix ACCEL
# RUN: llvm-dwarfdump -verify -debug-names %t.o

# CHECK: DW_AT_name    ("Bool")
# CHECK-NEXT: DW_AT_linkage_name    ("$SSbD")

# ACCEL-NOT: String: {{.*}} "$SSbD"
# ACCEL: String: {{.*}} "Bool"
# ACCEL-NOT: String: {{.*}} "$SSbD"

	.text
	.file	"local-vars.swift.gyb.tmp.s"
	.protected	main
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
.Lfunc_begin0:
.Ltmp0:
.Ltmp1:
.Lfunc_end0:
.Lfunc_begin1:
.Ltmp2:
.Ltmp3:
.Ltmp4:
.Lfunc_end1:
.Lfunc_begin2:
.Ltmp5:
.Ltmp6:
.Ltmp7:
.Lfunc_end2:
.Lfunc_begin3:
.Ltmp8:
.Ltmp9:
.Ltmp10:
.Lfunc_end3:
.Lfunc_begin4:
.Ltmp11:
.Ltmp12:
.Lfunc_end4:
.Lfunc_begin5:
.Ltmp13:
.LBB5_2:
.Ltmp14:
.Lfunc_end5:
.L__unnamed_1:
.L__unnamed_2:
.L__unnamed_3:
	.section	.swift1_autolink_entries,"a",@progbits
	.p2align	3
.L_swift1_autolink_entries:
	.asciz	"-lswiftSwiftOnoneSupport\000-lswiftCore"
	.size	.L_swift1_autolink_entries, 37

	.section	".linker-options","e",@llvm_linker_options
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"Swift version 4.2-dev effective-4.1.50 (LLVM a4b1bcaa22, Clang 257fa19143, Swift 836ad071bd)"
.Linfo_string1:
	.asciz	"/home/jonas/swift/build/Ninja-RelWithDebInfoAssert/swift-linux-x86_64/test-linux-x86_64/DebugInfo/Output/local-vars.swift.gyb.tmp.swift"
.Linfo_string2:
	.asciz	"/home/jonas/swift/build/Ninja-RelWithDebInfoAssert/swift-linux-x86_64/test-linux-x86_64/DebugInfo"
.Linfo_string3:
	.asciz	"main"
.Linfo_string4:
	.asciz	"/home/jonas/swift/build/Ninja-RelWithDebInfoAssert/swift-linux-x86_64/test-linux-x86_64/DebugInfo/Output"
.Linfo_string5:
	.asciz	"Swift"
.Linfo_string6:
	.asciz	"/home/jonas/swift/build/Ninja-RelWithDebInfoAssert/swift-linux-x86_64/lib/swift/linux/x86_64/Swift.swiftmodule"
.Linfo_string7:
	.asciz	"SwiftOnoneSupport"
.Linfo_string8:
	.asciz	"/home/jonas/swift/build/Ninja-RelWithDebInfoAssert/swift-linux-x86_64/lib/swift/linux/x86_64/SwiftOnoneSupport.swiftmodule"
.Linfo_string9:
	.asciz	"C"
.Linfo_string10:
	.asciz	"$S4main1CCD"
.Linfo_string11:
	.asciz	"init"
.Linfo_string12:
	.asciz	"$S4main1CCyACSicfc"
.Linfo_string13:
	.asciz	"isZero"
.Linfo_string14:
	.asciz	"$S4main1CC6isZeroSbyF"
.Linfo_string15:
	.asciz	"deinit"
.Linfo_string16:
	.asciz	"$S4main1CCfd"
.Linfo_string17:
	.asciz	"$S4main1CCfD"
.Linfo_string18:
	.asciz	"$S4main1CCMa"
.Linfo_string19:
	.asciz	"Int32"
.Linfo_string20:
	.asciz	"$Ss5Int32VD"
.Linfo_string21:
	.asciz	"Bool"
.Linfo_string22:
	.asciz	"$SSbD"
.Linfo_string23:
	.asciz	"$SBoD"
.Linfo_string24:
	.asciz	"$SytD"
.Linfo_string25:
	.asciz	"i"
.Linfo_string26:
	.asciz	"Int"
.Linfo_string27:
	.asciz	"$SSiD"
.Linfo_string28:
	.asciz	"self"
	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
	.ascii	"\345\177"
	.byte	11
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	2
	.byte	30
	.byte	1
	.byte	3
	.byte	14
	.ascii	"\200|"
	.byte	14
	.byte	0
	.byte	0
	.byte	3
	.byte	46
	.byte	0
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.byte	110
	.byte	14
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	4
	.byte	19
	.byte	1
	.byte	3
	.byte	14
	.byte	110
	.byte	14
	.byte	11
	.byte	11
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.ascii	"\346\177"
	.byte	11
	.byte	0
	.byte	0
	.byte	5
	.byte	46
	.byte	1
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.byte	110
	.byte	14
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	6
	.byte	5
	.byte	0
	.byte	2
	.byte	24
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	7
	.byte	5
	.byte	0
	.byte	2
	.byte	24
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	52
	.byte	25
	.byte	0
	.byte	0
	.byte	8
	.byte	46
	.byte	0
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.byte	110
	.byte	14
	.byte	52
	.byte	25
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	9
	.byte	58
	.byte	0
	.byte	24
	.byte	19
	.byte	0
	.byte	0
	.byte	10
	.byte	19
	.byte	0
	.byte	3
	.byte	14
	.byte	110
	.byte	14
	.byte	11
	.byte	11
	.ascii	"\346\177"
	.byte	11
	.byte	0
	.byte	0
	.byte	11
	.byte	30
	.byte	0
	.byte	3
	.byte	14
	.ascii	"\200|"
	.byte	14
	.byte	0
	.byte	0
	.byte	12
	.byte	15
	.byte	0
	.byte	3
	.byte	14
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	385
	.short	4
	.long	.debug_abbrev
	.byte	8
	.byte	1
	.long	.Linfo_string0
	.short	30
	.long	.Linfo_string1
	.long	.Lline_table_start0
	.long	.Linfo_string2
	.byte	4
	.quad	.Lfunc_begin0
	.long	.Lfunc_end5-.Lfunc_begin0
	.byte	2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.byte	3
	.quad	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	86
	.long	.Linfo_string3
	.long	.Linfo_string3
	.byte	1
	.byte	1
	.long	319

	.byte	4
	.long	.Linfo_string9
	.long	.Linfo_string10
	.byte	8
	.byte	1
	.byte	9
	.byte	30
	.byte	5
	.quad	.Lfunc_begin1
	.long	.Lfunc_end1-.Lfunc_begin1
	.byte	1
	.byte	86
	.long	.Linfo_string12
	.long	.Linfo_string11
	.byte	1
	.byte	11
	.long	81

	.byte	6
	.byte	2
	.byte	145
	.byte	120
	.long	.Linfo_string25
	.byte	1
	.byte	11
	.long	341
	.byte	7
	.byte	2
	.byte	145
	.byte	112
	.long	.Linfo_string28
	.byte	1
	.byte	11
	.long	81

	.byte	0
	.byte	5
	.quad	.Lfunc_begin2
	.long	.Lfunc_end2-.Lfunc_begin2
	.byte	1
	.byte	86
	.long	.Linfo_string14
	.long	.Linfo_string13
	.byte	1
	.byte	12
	.long	330

	.byte	7
	.byte	2
	.byte	145
	.byte	112
	.long	.Linfo_string28
	.byte	1
	.byte	12
	.long	81

	.byte	0
	.byte	5
	.quad	.Lfunc_begin3
	.long	.Lfunc_end3-.Lfunc_begin3
	.byte	1
	.byte	86
	.long	.Linfo_string16
	.long	.Linfo_string15
	.byte	1
	.byte	9
	.long	372

	.byte	7
	.byte	2
	.byte	145
	.byte	120
	.long	.Linfo_string28
	.byte	1
	.byte	9
	.long	81

	.byte	0
	.byte	5
	.quad	.Lfunc_begin4
	.long	.Lfunc_end4-.Lfunc_begin4
	.byte	1
	.byte	86
	.long	.Linfo_string17
	.long	.Linfo_string15
	.byte	1
	.byte	9
	.long	377

	.byte	7
	.byte	2
	.byte	145
	.byte	120
	.long	.Linfo_string28
	.byte	1
	.byte	9
	.long	81

	.byte	0
	.byte	0
	.byte	8
	.quad	.Lfunc_begin5
	.long	.Lfunc_end5-.Lfunc_begin5
	.byte	1
	.byte	86
	.long	.Linfo_string18


	.byte	0
	.byte	9
	.long	43
	.byte	2
	.long	.Linfo_string5
	.long	.Linfo_string6
	.byte	10
	.long	.Linfo_string19
	.long	.Linfo_string20
	.byte	4
	.byte	30
	.byte	10
	.long	.Linfo_string21
	.long	.Linfo_string22
	.byte	1
	.byte	30
	.byte	10
	.long	.Linfo_string26
	.long	.Linfo_string27
	.byte	8
	.byte	30
	.byte	0
	.byte	9
	.long	310
	.byte	11
	.long	.Linfo_string7
	.long	.Linfo_string8
	.byte	9
	.long	358
	.byte	12
	.long	.Linfo_string23
	.byte	10
	.long	.Linfo_string24
	.long	.Linfo_string24
	.byte	0
	.byte	30
	.byte	0
	.section	.debug_ranges,"",@progbits
	.section	.debug_macinfo,"",@progbits
	.byte	0
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0
.Lnames_start0:
	.short	5
	.short	0
	.long	1
	.long	0
	.long	0
	.long	14
	.long	15
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0
	.long	8
	.ascii	"LLVM0700"
	.long	.Lcu_begin0
	.long	0
	.long	0
	.long	1
	.long	0
	.long	2
	.long	0
	.long	0
	.long	4
	.long	6
	.long	7
	.long	11
	.long	14
	.long	15
	.long	0
	.long	-125696958
	.long	-1434607370
	.long	-1434607370
	.long	87184321
	.long	2090120081
	.long	-1434607142
	.long	181113837
	.long	262755061
	.long	2090370361
	.long	-1008003439
	.long	193495088
	.long	2090499946
	.long	-1294887406
	.long	181088625
	.long	177672
	.long	.Linfo_string15
	.long	.Linfo_string16
	.long	.Linfo_string17
	.long	.Linfo_string13
	.long	.Linfo_string21
	.long	.Linfo_string18
	.long	.Linfo_string24
	.long	.Linfo_string19
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string26
	.long	.Linfo_string3
	.long	.Linfo_string14
	.long	.Linfo_string23
	.long	.Linfo_string9
	.long	.Lnames14-.Lnames_entries0
	.long	.Lnames9-.Lnames_entries0
	.long	.Lnames10-.Lnames_entries0
	.long	.Lnames13-.Lnames_entries0
	.long	.Lnames7-.Lnames_entries0
	.long	.Lnames12-.Lnames_entries0
	.long	.Lnames2-.Lnames_entries0
	.long	.Lnames8-.Lnames_entries0
	.long	.Lnames11-.Lnames_entries0
	.long	.Lnames5-.Lnames_entries0
	.long	.Lnames3-.Lnames_entries0
	.long	.Lnames1-.Lnames_entries0
	.long	.Lnames6-.Lnames_entries0
	.long	.Lnames4-.Lnames_entries0
	.long	.Lnames0-.Lnames_entries0
.Lnames_abbrev_start0:
	.byte	46
	.byte	46
	.byte	3
	.byte	19
	.byte	0
	.byte	0
	.byte	15
	.byte	15
	.byte	3
	.byte	19
	.byte	0
	.byte	0
	.byte	19
	.byte	19
	.byte	3
	.byte	19
	.byte	0
	.byte	0
	.byte	0
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames14:
	.byte	46
	.long	196
	.byte	46
	.long	240
	.long	0
.Lnames9:
	.byte	46
	.long	196
	.long	0
.Lnames10:
	.byte	46
	.long	240
	.long	0
.Lnames13:
	.byte	46
	.long	152
	.long	0
.Lnames7:
	.byte	19
	.long	330
	.long	0
.Lnames12:
	.byte	46
	.long	285
	.long	0
.Lnames2:
	.byte	19
	.long	377
	.long	0
.Lnames8:
	.byte	19
	.long	319
	.long	0
.Lnames11:
	.byte	46
	.long	94
	.long	0
.Lnames5:
	.byte	46
	.long	94
	.long	0
.Lnames3:
	.byte	19
	.long	341
	.long	0
.Lnames1:
	.byte	46
	.long	52
	.long	0
.Lnames6:
	.byte	46
	.long	152
	.long	0
.Lnames4:
	.byte	15
	.long	372
	.long	0
.Lnames0:
	.byte	19
	.long	81
	.long	0
	.p2align	2
.Lnames_end0:

	.globl	$S4main1CCN
	.protected	$S4main1CCN
.set $S4main1CCN, ($S4main1CCMf)+16
	.section	.debug_line,"",@progbits
.Lline_table_start0:
