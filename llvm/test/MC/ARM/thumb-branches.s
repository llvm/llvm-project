@ RUN: not llvm-mc %s -triple thumbv7-linux-gnueabi -filetype=obj -o /dev/null 2>&1 | FileCheck %s
@ RUN: not llvm-mc %s -triple thumbv8-m.baseline-none-eabi -filetype=obj -o /dev/null 2>&1 | FileCheck %s
@ RUN: not llvm-mc %s -triple thumbv8-m.mainline-none-eabi -filetype=obj -o /dev/null 2>&1 | FileCheck %s
@ RUN: not llvm-mc %s -triple thumbv6m-none-eabi -filetype=obj -o /dev/null 2>&1 | FileCheck %s
@ RUN: not llvm-mc %s -triple thumbv5-linux-gnueabi -filetype=obj -o /dev/null 2>&1 | FileCheck -check-prefix=CHECKSHORT %s

// Thumb BL has range +- 4 Megabytes if CPU does not support Thumb2 or does not
// have v8-M baseline ops, it is +- 16 Megabytes otherwise.

        .code 16
        bl      shortend
        .space 0x3fffff
shortend:
// CHECKSHORT-NOT: error
// CHECKSHORT: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl      shortend2
        .space 0x400000
shortend2:

// CHECKSHORT: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl      end
        .space 0xffffff
end:
        bl      end2
        .space 0xffffff
        .global end2
end2:

        bl      end3
        .space 0x1000000
        .global end3
end3:

// CHECK-NOT: error
// CHECKSHORT-NOT: error
// CHECKSHORT: [[@LINE+2]]:{{[0-9]}}: error: Relocation out of range
// CHECK: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl      end4
        .space 0x1000000
end4:

shortstart1:
        .space 0x3ffffc
        bl shortstart1

shortstart2:
        .space 0x400000
// CHECKSHORT: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl shortstart2

start1:
        .space 0xfffffc
// CHECKSHORT: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl start1

        .global start2
start2:
        .space 0xfffffc
        bl start2

        .global start3
start3:
        .space 0xfffffd
        bl start3

// CHECK-NOT: error
start4:
        .space 0xfffffd
// CHECK: [[@LINE+2]]:{{[0-9]}}: error: Relocation out of range
// CHECKSHORT: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl start4
