// RUN: llvm-mc -filetype=asm -triple powerpc-ibm-aix-xcoff %s | FileCheck %s

// CHECK:             lwarx 3, 0, 4
// CHECK-NEXT:        cmpw    5, 3
// CHECK-NEXT:        bne-    0, L..tmp1
// CHECK-NEXT:        stwcx. 6, 0, 4
// CHECK-NEXT:        bne-    0, L..tmp0
// CHECK-NEXT:L..tmp1:


        #APP
1:
        lwarx 3, 0, 4
        cmpw    5, 3
        bne-    2f
        stwcx. 6, 0, 4
        bne-    1b
2:

        #NO_APP

