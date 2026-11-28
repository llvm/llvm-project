## Basic checks for `hmx` attribute and `-mhmx` flag

# RUN: llvm-mc -triple=hexagon -mhmx -filetype=obj %s | \
# RUN:   llvm-objdump -d --mattr=+hmx - | FileCheck %s

# RUN: llvm-mc -triple=hexagon -mattr=+hmx -filetype=obj %s | \
# RUN:   llvm-objdump -d --mattr=+hmx - | FileCheck %s

# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | \
# RUN:   FileCheck --check-prefix=CHECK-NOHMX --implicit-check-not='error:' %s

# RUN: llvm-mc -triple=hexagon -mhmx -filetype=obj %s | \
# RUN:   llvm-objdump -d - | \
# RUN:   FileCheck --check-prefix=CHECK-DIS-NOHMX %s

mxclracc.hf

# CHECK: { mxclracc.hf }

# CHECK-NOHMX: hmx-support.s:[[#@LINE-4]]:1: error: invalid instruction

# CHECK-DIS-NOHMX:       unknown
