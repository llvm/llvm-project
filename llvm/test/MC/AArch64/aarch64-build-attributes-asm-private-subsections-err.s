// RUN: not llvm-mc -triple=aarch64 %s 2>&1 | FileCheck --check-prefix=ERR %s

.aeabi_subsection private_subsection, optional, uleb128

.aeabi_subsection private_subsection, required, uleb128
// ERR: error: optionality mismatch! subsection 'private_subsection' already exists with optionality defined as 'optional' and not 'required'
// ERR-NEXT: .aeabi_subsection private_subsection, required, uleb128

.aeabi_subsection private_subsection, optional, ntbs
// ERR: error: type mismatch! subsection 'private_subsection' already exists with type defined as 'uleb128' and not 'ntbs'
// ERR-NEXT: .aeabi_subsection private_subsection, optional, ntbs

.aeabi_subsection private_subsection_1, optional, ntbs
.aeabi_attribute 324, 1
// ERR: error: active subsection type is NTBS (string), found ULEB128 (unsigned)
// ERR-NEXT: .aeabi_attribute 324, 1

.aeabi_subsection foo, optional, uleb128
.aeabi_subsection bar, optional, uleb128
.aeabi_subsection foo, required, uleb128
// ERR: error: optionality mismatch! subsection 'foo' already exists with optionality defined as 'optional' and not 'required'
// ERR-NEXT: .aeabi_subsection foo, required, uleb128

.aeabi_subsection goo, optional, ntbs
.aeabi_subsection zar, optional, ntbs
.aeabi_subsection goo, optional, uleb128
// ERR: error: type mismatch! subsection 'goo' already exists with type defined as 'ntbs' and not 'uleb128'
// ERR-NEXT: .aeabi_subsection goo, optional, uleb128
