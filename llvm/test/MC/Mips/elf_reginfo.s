# These *MUST* match the output of gas compiled with the same triple and
# corresponding options (-mabi=64 -> -mattr=+n64 for example).

# RUN: llvm-mc -filetype=obj -triple=mips64el-linux -target-abi n64 %s -o - \
# RUN: | llvm-readobj -S | FileCheck --check-prefix=CHECK_64 %s
# RUN: llvm-mc -filetype=obj -triple=mipsel %s -target-abi n32 -o - \
# RUN: | llvm-readobj -S | FileCheck --check-prefix=CHECK_32 %s

# Check for register information sections.
#

# Check that the appropriate relocations were created.

# check for .MIPS.options
# CHECK_64:      Sections [
# CHECK_64:        Section {
# CHECK_64-LABEL:    Name: .MIPS.options
# CHECK_64-NEXT:     Type: SHT_MIPS_OPTIONS
# CHECK_64-NEXT:     Flags [ (0x8000002)
# CHECK_64:          AddressAlignment: 8
# CHECK_64:          EntrySize: 1
# CHECK_64-LABEL:  }

# check for .reginfo
# CHECK_32:      Sections [
# CHECK_32:        Section {
# CHECK_32-LABEL:    Name: .reginfo
# CHECK_32-NEXT:     Type:  SHT_MIPS_REGINFO
# CHECK_32-NEXT:     Flags [ (0x2)
# CHECK_32:          AddressAlignment: 8
# CHECK_32:          EntrySize: 24
# CHECK_32-LABEL:  }
