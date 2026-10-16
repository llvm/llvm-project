# RUN: llvm-mc --triple=loongarch32 --loongarch-no-aliases %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch32 -M no-aliases %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch32 --filetype=obj %s -o %t.32
# RUN: llvm-objdump -d -M no-aliases %t.32 | FileCheck %s

# RUN: llvm-mc --triple=loongarch64 --loongarch-no-aliases %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch64 -M no-aliases %s \
# RUN:     | FileCheck %s
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s -o %t.64
# RUN: llvm-objdump -d -M no-aliases %t.64 | FileCheck %s

# Also test passing multiple disassembly options at once.
# RUN: llvm-objdump -d -M no-aliases,numeric %t.64 | FileCheck --check-prefix=CHECK-NUMERIC %s

foo:
    # CHECK:              or $a0, $r21, $zero
    # CHECK-NEXT:         jirl $zero, $ra, 0
    # CHECK-NUMERIC:      or $r4, $r21, $r0
    # CHECK-NUMERIC-NEXT: jirl $r0, $r1, 0
    move $a0, $r21
    ret
