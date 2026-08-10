# Test object to verify that dwarfdump does not go into infinite recursion due
# to trying to print fully resolved name.
# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump --debug-info=0x62 %t.o | FileCheck %s

# CHECK: DW_TAG_formal_parameter
# CHECK-NEXT: DW_AT_type
# CHECK-SAME: "t2 *"

# gcc -g -gdwarf-4 -std=gnu++17 -c -gz=none -S reproSmall.cpp -o reproSmall.s
# struct t1;
# void f1() {
#   using t2 = t1;
#   void (t2::* __fn)();
# }

.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x88
	.value	0x4
	.long	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.long	.LASF0
	.byte	0x4
	.long	.LASF1
	.long	.LASF2
	.quad	0
	.quad	0
	.long	.Ldebug_line0
	.uleb128 0x2
	.string	"f1"
	.byte	0x1
	.byte	0x3
	.byte	0x6
	.long	.LASF3
	.quad	0
	.quad	0
	.uleb128 0x1
	.byte	0x9c
	.long	0x87
	.uleb128 0x3
	.string	"t2"
	.byte	0x1
	.byte	0x4
	.byte	0x9
	.long	0x87
	.uleb128 0x4
	.long	0x62
	.long	0x6e
	.uleb128 0x5
	.long	0x67
	.uleb128 0x6
	.byte	0x8
	.long	0x4e
	.byte	0
	.uleb128 0x7
	.long	0x87
	.long	0x59
	.uleb128 0x8
	.long	.LASF4
	.byte	0x1
	.byte	0x5
	.byte	0xf
	.long	0x6e
	.uleb128 0x2
	.byte	0x91
	.sleb128 -32
	.byte	0
	.uleb128 0x9
	.string	"t1"
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x1b
	.uleb128 0xe
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x15
	.byte	0x1
	.uleb128 0x64
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0x5
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x34
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x1f
	.byte	0
	.uleb128 0x1d
	.uleb128 0x13
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0x13
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_aranges,"",@progbits
	.long	0x2c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF0:
	.string	"GNU C++17 11.x -mtune=generic -march=x86-64 -g -gdwarf-4 -gz=none -std=gnu++17"
.LASF4:
	.string	"__fn"
.LASF2:
	.string	"."
.LASF1:
	.string	"reproSmall.cpp"
.LASF3:
	.string	"_Z2f1v"
	.ident	"GCC: (GNU) 11.x"
