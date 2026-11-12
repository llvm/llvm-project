# RUN: llvm-mc %S/brief.s -filetype obj -triple x86_64-apple-darwin -o %t.o
# RUN: llvm-dwarfdump --verify %t.o | FileCheck %s
# RUN: llvm-dwarfdump --verify --quiet %t.o | FileCheck %s --check-prefix=QUIET --allow-empty

# RUN: llvm-mc %S/empty-CU.s -filetype obj -triple x86_64-apple-darwin -o %t-error.o
# RUN: not llvm-dwarfdump --verify %t-error.o | FileCheck %s
# RUN: not llvm-dwarfdump --verify --quiet %t-error.o | FileCheck %s --check-prefix=QUIET --allow-empty

# CHECK: {{.}}
# QUIET-NOT: {{.}}
