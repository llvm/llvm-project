
# RUN: llvm-mc -triple s390x-unknown-unknown -mcpu=zEC12 --show-encoding %s | FileCheck %s

# RUN: llvm-mc -triple s390x-unknown-unknown -mcpu=zEC12 -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-REL

# CHECK: bpp 12, branch, 0                      # encoding: [0xc7,0xc0,0x00,0x00,A,A]
# CHECK:                                        # fixup A - offset: 4, value: branch+4, kind: FK_390_PC16DBL
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_PC16DBL branch 0x4
	.align 16
        bpp 12, branch, 0

# CHECK: bpp 12, branch@PLT, 0                  # encoding: [0xc7,0xc0,0x00,0x00,A,A]
# CHECK:                                        # fixup A - offset: 4, value: branch@PLT+4, kind: FK_390_PC16DBL
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_PLT16DBL branch 0x4
	.align 16
        bpp 12, branch@plt, 0

# CHECK: bprp 12, branch, target                # encoding: [0xc5,0b1100AAAA,A,B,B,B]
# CHECK-NEXT:                                   # fixup A - offset: 1, value: branch+1, kind: FK_390_PC12DBL
# CHECK-NEXT:                                   # fixup B - offset: 3, value: target+3, kind: FK_390_PC24DBL
# CHECK-REL:                                    0x{{[0-9A-F]*1}} R_390_PC12DBL branch 0x1
# CHECK-REL:                                    0x{{[0-9A-F]*3}} R_390_PC24DBL target 0x3
	.align 16
	bprp 12, branch, target

# CHECK: bprp 12, branch@PLT, target@PLT        # encoding: [0xc5,0b1100AAAA,A,B,B,B]
# CHECK-NEXT:                                   # fixup A - offset: 1, value: branch@PLT+1, kind: FK_390_PC12DBL
# CHECK-NEXT:                                   # fixup B - offset: 3, value: target@PLT+3, kind: FK_390_PC24DBL
# CHECK-REL:                                    0x{{[0-9A-F]*1}} R_390_PLT12DBL branch 0x1
# CHECK-REL:                                    0x{{[0-9A-F]*3}} R_390_PLT24DBL target 0x3
	.align 16
	bprp 12, branch@plt, target@plt

