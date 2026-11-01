# RUN: llvm-mc %s -triple=csky | FileCheck %s
# RUN: llvm-mc %s -triple=csky -mcpu=ck860 | FileCheck %s --check-prefix=CK860
# RUN: llvm-mc %s -triple=csky -csky-add-build-attributes=false | FileCheck %s --check-prefix=OFF

nop

# CHECK: .csky_attribute 4, "ck810"
# CHECK-NEXT: .csky_attribute 5, "ck810"

# CK860: .csky_attribute 4, "ck860"
# CK860-NEXT: .csky_attribute 5, "ck860"

# OFF-NOT: .csky_attribute
