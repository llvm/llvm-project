// RUN: llvm-mc -triple amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=ASM %s

// ASM: .set alignto_zero_eight, 0
// ASM: .set alignto_one_eight, 8
// ASM: .set alignto_five_eight, 8
// ASM: .set alignto_seven_eight, 8
// ASM: .set alignto_eight_eight, 8
// ASM: .set alignto_ten_eight, 16

.set alignto_zero_eight, alignto(0, 8)
.set alignto_one_eight, alignto(1, 8)
.set alignto_five_eight, alignto(5, 8)
.set alignto_seven_eight, alignto(7, 8)
.set alignto_eight_eight, alignto(8, 8)
.set alignto_ten_eight, alignto(10, 8)
