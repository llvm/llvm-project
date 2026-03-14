# RUN: llvm-as %S/Inputs/f.ll -o %t.f.bc
# RUN: llvm-as %S/Inputs/g.ll -o %t.g.bc
# RUN: echo "This is not a bitcode file" > %t.not_bitcode.txt
# RUN: llvm-ar cr %t.a %t.f.bc %t.not_bitcode.txt %t.g.bc
# RUN: llvm-ar cr --format=gnu %t.empty.lib
# RUN: llvm-link -ignore-non-bitcode %t.a %t.empty.lib -o %t.linked.bc 2>&1 | FileCheck --check-prefix CHECK_IGNORE_NON_BITCODE %s
# RUN: not llvm-link %t.a %t.empty.lib -o %t.linked2.bc 2>&1 | FileCheck --check-prefix CHECK_ERROR_BITCODE %s

# CHECK_ERROR_BITCODE: error: member of archive is not a bitcode file
# CHECK_IGNORE_NON_BITCODE-NOT: is not a bitcode file
