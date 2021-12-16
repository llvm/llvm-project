# RUN: llvm-mc %s -filetype obj -triple x86_64-linux-gnu -o - \
# RUN: | not llvm-dwarfdump -v -verify - \
# RUN: | FileCheck %s --implicit-check-not=error --implicit-check-not=warning

# CHECK: Verifying dwo Units...
# CHECK: error: Compilation unit root DIE is not a unit DIE: DW_TAG_null.
# CHECK: error: Compilation unit type (DW_UT_split_compile) and root DIE (DW_TAG_null) do not match.
# CHECK: Errors detected
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	5527374834836270265
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end1:
