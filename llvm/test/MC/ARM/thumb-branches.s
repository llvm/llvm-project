@ RUN: not llvm-mc %s -triple thumbv5-linux-gnueabi -filetype=obj -o /dev/null 2>&1 | FileCheck %s

        .code 16

        bl      end
        .space 0x1ffffff
end:

        bl      end2
        .space 0x1ffffff
        .global end2
end2:

        bl      end3
        .space 0x2000000
        .global end3
end3:

// CHECK-NOT: error
// CHECK: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl      end4
// CHECK-NOT: error
        .space 0x2000000
end4:

start1:
        .space 0x1fffffc
        bl start1

        .global start2
start2:
        .space 0x1fffffc
        bl start2

        .global start3
start3:
        .space 0x1fffffd
        bl start3

start4:
        .space 0x1fffffd
// CHECK: [[@LINE+1]]:{{[0-9]}}: error: Relocation out of range
        bl start4
