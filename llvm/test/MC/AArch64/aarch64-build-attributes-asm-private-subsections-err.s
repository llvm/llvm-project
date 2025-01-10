// RUN: not llvm-mc -triple=aarch64 %s -o %t > %t.out 2>&1
// RUN: FileCheck --input-file=%t.out --check-prefix=ERR %s 

.aeabi_subsection private_subsection, optional, uleb128

.aeabi_subsection private_subsection, required, uleb128
// ERR: error: Optinality mismatch! Subsection private_subsection allready exists with optinality defined as '1' and not '0'! (0: required, 1: optional)
// ERR-NEXT: .aeabi_subsection private_subsection, required, uleb128

.aeabi_subsection private_subsection, optional, ntbs
// ERR: error: Type mismatch! Subsection private_subsection allready exists with Type defined as '0' and not '1'! (0: uleb128, 1: ntbs)
// ERR-NEXT: .aeabi_subsection private_subsection, optional, ntbs