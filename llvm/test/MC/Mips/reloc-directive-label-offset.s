# RUN: llvm-mc -triple mips-unknown-linux %s -show-encoding -target-abi=o32 \
# RUN:     | FileCheck --check-prefixes=ASM,ASM-32 %s
# RUN: llvm-mc -triple mips64-unknown-linux %s -show-encoding -target-abi=n32 \
# RUN:     | FileCheck --check-prefixes=ASM,ASM-64 %s
# RUN: llvm-mc -triple mips64-unknown-linux %s -show-encoding -target-abi=n64 \
# RUN:     | FileCheck --check-prefixes=ASM,ASM-64 %s
# RUN: llvm-mc -triple mips-unknown-linux %s -show-encoding -target-abi=o32 \
# RUN:     -filetype=obj | llvm-readobj -r | FileCheck -check-prefix=OBJ-O32 %s
# RUN: llvm-mc -triple mips64-unknown-linux %s -show-encoding -target-abi=n32 \
# RUN:     -filetype=obj | llvm-readobj -r | FileCheck -check-prefix=OBJ-N32 %s
# RUN: llvm-mc -triple mips64-unknown-linux %s -show-encoding -target-abi=n64 \
# RUN:     -filetype=obj | llvm-readobj -r | FileCheck -check-prefix=OBJ-N64 %s

  .text
foo: # ASM-LABEL: foo:
  nop
1:
  nop
  .reloc 1b, R_MIPS_NONE, foo       # ASM-32: .reloc ($tmp0), R_MIPS_NONE, foo
                                    # ASM-64: .reloc .Ltmp0, R_MIPS_NONE, foo
  nop
  .reloc 1f, R_MIPS_32, foo         # ASM-32: .reloc ($tmp1), R_MIPS_32, foo
                                    # ASM-64: .reloc .Ltmp1, R_MIPS_32, foo
1:
  nop
  .reloc 1f, R_MIPS_CALL16, foo     # ASM-32: .reloc ($tmp2), R_MIPS_CALL16, foo
                                    # ASM-64: .reloc .Ltmp2, R_MIPS_CALL16, foo
1:
  nop
  .reloc 2f, R_MIPS_GOT_DISP, foo   # ASM-32: .reloc ($tmp3), R_MIPS_GOT_DISP, foo
                                    # ASM-64: .reloc .Ltmp3, R_MIPS_GOT_DISP, foo
  nop

  .reloc 3f, R_MIPS_GOT_PAGE, foo   # ASM-32: .reloc ($tmp4), R_MIPS_GOT_PAGE, foo
                                    # ASM-64: .reloc .Ltmp4, R_MIPS_GOT_PAGE, foo
  nop
bar:
  nop
2:
  nop
3:
  nop
  .reloc bar, R_MIPS_GOT_OFST, foo  # ASM: .reloc bar, R_MIPS_GOT_OFST, foo
  nop
  .reloc foo, R_MIPS_32, foo        # ASM: .reloc foo, R_MIPS_32, foo
  nop
1:
  nop

# OBJ-O32-LABEL: Relocations [
# OBJ-O32:           0x0 R_MIPS_32 .text 0x0
# OBJ-O32-NEXT:      0x4 R_MIPS_NONE .text 0x0
# OBJ-O32-NEXT:      0xC R_MIPS_32 .text 0x0
# OBJ-O32-NEXT:      0x10 R_MIPS_CALL16 foo 0x0
# OBJ-O32-NEXT:      0x1C R_MIPS_GOT_OFST .text 0x0
# OBJ-O32-NEXT:      0x20 R_MIPS_GOT_DISP foo 0x0
# OBJ-O32-NEXT:      0x24 R_MIPS_GOT_PAGE .text 0x0

# OBJ-N32-LABEL: Relocations [
# OBJ-N32:           0x4 R_MIPS_NONE .text 0x0
# OBJ-N32-NEXT:      0x1C R_MIPS_GOT_OFST .text 0x0
# OBJ-N32-NEXT:      0x0 R_MIPS_32 .text 0x0
# OBJ-N32-NEXT:      0xC R_MIPS_32 .text 0x0
# OBJ-N32-NEXT:      0x10 R_MIPS_CALL16 foo 0x0
# OBJ-N32-NEXT:      0x20 R_MIPS_GOT_DISP foo 0x0
# OBJ-N32-NEXT:      0x24 R_MIPS_GOT_PAGE .text 0x0

# OBJ-N64-LABEL: Relocations [
# OBJ-N64:           0x4 R_MIPS_NONE/R_MIPS_NONE/R_MIPS_NONE .text 0x0
# OBJ-N64-NEXT:      0x1C R_MIPS_GOT_OFST/R_MIPS_NONE/R_MIPS_NONE .text 0x0
# OBJ-N64-NEXT:      0x0 R_MIPS_32/R_MIPS_NONE/R_MIPS_NONE .text 0x0
# OBJ-N64-NEXT:      0xC R_MIPS_32/R_MIPS_NONE/R_MIPS_NONE .text 0x0
# OBJ-N64-NEXT:      0x10 R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE foo 0x0
# OBJ-N64-NEXT:      0x20 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE foo 0x0
# OBJ-N64-NEXT:      0x24 R_MIPS_GOT_PAGE/R_MIPS_NONE/R_MIPS_NONE .text 0x0
