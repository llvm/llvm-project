// REQUIRES: zlib
// Check zlib-gnu style
// RUN: llvm-mc -filetype=obj -compress-debug-sections=zlib-gnu -triple x86_64-pc-linux-gnu < %s -o %t
// RUN: llvm-objdump -s %t | FileCheck --check-prefix=CHECK-GNU-STYLE %s
// RUN: llvm-dwarfdump -debug-str %t | FileCheck --check-prefix=STR %s
// RUN: llvm-mc -filetype=obj -compress-debug-sections=zlib-gnu -triple i386-pc-linux-gnu < %s \
// RUN:     | llvm-readobj -symbols - | FileCheck --check-prefix=386-SYMBOLS-GNU %s

// Check zlib style
// RUN: llvm-mc -filetype=obj -compress-debug-sections=zlib -triple x86_64-pc-linux-gnu < %s -o %t
// RUN: llvm-objdump -s %t | FileCheck --check-prefix=CHECK-ZLIB-STYLE %s
// RUN: llvm-dwarfdump -debug-str %t | FileCheck --check-prefix=STR %s
// RUN: llvm-mc -filetype=obj -compress-debug-sections=zlib -triple i386-pc-linux-gnu < %s \
// RUN:     | llvm-readobj -symbols - | FileCheck --check-prefix=386-SYMBOLS-ZLIB %s
// RUN: llvm-readobj -sections %t | FileCheck --check-prefix=ZLIB-STYLE-FLAGS %s

// Don't compress small sections, such as this simple debug_abbrev example
// CHECK-GNU-STYLE: Contents of section .debug_abbrev:
// CHECK-GNU-STYLE-NOT: ZLIB
// CHECK-GNU-STYLE-NOT: Contents of

// CHECK-GNU-STYLE: Contents of section .debug_info:

// CHECK-GNU-STYLE: Contents of section .zdebug_str:
// Check for the 'ZLIB' file magic at the start of the section only
// CHECK-GNU-STYLE-NEXT: ZLIB
// CHECK-GNU-STYLE-NOT: ZLIB
// FIXME: Handle compressing alignment fragments to support compressing debug_frame
// CHECK-GNU-STYLE: Contents of section .debug_frame:
// CHECK-GNU-STYLE-NOT: ZLIB
// CHECK-GNU-STYLE: Contents of

// Decompress one valid dwarf section just to check that this roundtrips,
// we use .zdebug_str section for that
// STR: perfectly compressable data sample *****************************************

// In x86 32 bit named symbols are used for temporary symbols in merge
// sections, so make sure we handle symbols inside compressed sections
// 386-SYMBOLS-GNU: Name: .Linfo_string0
// 386-SYMBOLS-GNU-NOT: }
// 386-SYMBOLS-GNU: Section: .zdebug_str

// Now check the zlib style output:

// Don't compress small sections, such as this simple debug_abbrev example
// CHECK-ZLIB-STYLE: Contents of section .debug_abbrev:
// CHECK-ZLIB-STYLE-NOT: ZLIB
// CHECK-ZLIB-STYLE-NOT: Contents of
// CHECK-ZLIB-STYLE: Contents of section .debug_info:
// FIXME: Handle compressing alignment fragments to support compressing debug_frame
// CHECK-ZLIB-STYLE: Contents of section .debug_frame:
// CHECK-ZLIB-STYLE-NOT: ZLIB
// CHECK-ZLIB-STYLE: Contents of

// Check that debug_line section was not renamed, so it is
// zlib-style, not zlib-gnu one. Check that SHF_COMPRESSED was set.
// ZLIB-STYLE-FLAGS:      Section {
// ZLIB-STYLE-FLAGS:        Index:
// ZLIB-STYLE-FLAGS:        Name: .debug_str
// ZLIB-STYLE-FLAGS-NEXT:   Type: SHT_PROGBITS
// ZLIB-STYLE-FLAGS-NEXT:   Flags [
// ZLIB-STYLE-FLAGS-NEXT:     SHF_COMPRESSED

// 386-SYMBOLS-ZLIB: Name: .Linfo_string0
// 386-SYMBOLS-ZLIB-NOT: }
// 386-SYMBOLS-ZLIB: Section: .debug_str

	.section	.debug_line,"",@progbits

	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.section	.debug_info,"",@progbits
	.long	12                      # Length of Unit
	.short	4                       # DWARF version number
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_comp_dir

	.text
foo:
	.cfi_startproc
	.file 1 "Driver.ii"
# pad out the line table to make sure it's big enough to warrant compression
	.loc 1 2 0
        nop
	.loc 1 3 0
        nop
	.loc 1 4 0
        nop
	.loc 1 5 0
        nop
	.loc 1 6 0
        nop
	.loc 1 7 0
        nop
	.loc 1 8 0
        nop
	.cfi_endproc
	.cfi_sections .debug_frame

# Below is the section we will use to check that after compression with llvm-mc,
# llvm-dwarfdump tool will be able to decompress data back and dump it. Data sample
# should be compressable enough, so it is filled with some amount of equal symbols at the end
	.section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "perfectly compressable data sample *****************************************"
