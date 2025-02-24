# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --arch-specific - 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: unable to dump attributes from the Unknown section with index 3: unknown public AArch64 build attribute subsection name at offset: 5

.aeabi_subsection aeabi_a, optional, uleb128
