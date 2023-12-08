# RUN: llvm-mc -triple=hexagon -filetype=asm %s -o - | FileCheck %s

# Verify empty packets aren't printed
barrier
{}
barrier
# CHECK: {
# CHECK-NEXT: barrier
# CHECK-NEXT: }
# CHECK-NOT: }
# CHECK: {
# CHECK-NEXT: barrier
# CHECK-NEXT: }


