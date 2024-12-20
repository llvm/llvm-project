// RUN: not llvm-mc -triple=aarch64 %s -o %t > %t.out 2>&1
// RUN: FileCheck --input-file=%t.out --check-prefix=ERR %s 

.aeabi_subsection aeabi_pauthabi, optional, uleb128
// ERR: error: aeabi_pauthabi must be marked as required
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, optional, uleb128

.aeabi_subsection aeabi_pauthabi, required, ntbs
// ERR: error: aeabi_pauthabi must be marked as ULEB128
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required, ntbs

.aeabi_subsection aeabi_feature_and_bits, required, uleb128
// ERR: error: aeabi_feature_and_bits must be marked as optional
// ERR-NEXT: .aeabi_subsection aeabi_feature_and_bits, required, uleb128

.aeabi_subsection aeabi_feature_and_bits, optional, ntbs
// ERR: error: aeabi_feature_and_bits must be marked as ULEB128
// ERR-NEXT: .aeabi_subsection aeabi_feature_and_bits, optional, ntbs

.aeabi_subsection a, required, uleb128
// ERR: error: unknown AArch64 build attributes subsection: a
// ERR-NEXT: .aeabi_subsection a, required, uleb128

.aeabi_subsection 1, required, uleb128
// ERR: error: Expecting subsection name
// ERR-NEXT: .aeabi_subsection 1, required, uleb128

.aeabi_subsection , required, uleb128
// ERR: error: Expecting subsection name
// ERR-NEXT: .aeabi_subsection , required, uleb128

.aeabi_subsection required, uleb128
// ERR: error: unknown AArch64 build attributes subsection: required
// ERR-NEXT: .aeabi_subsection required, uleb128

.aeabi_subsection aeabi_pauthabi, a, uleb128
// ERR: error: unknown AArch64 build attributes optionality, expecting required|optional: a
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, a, uleb128

.aeabi_subsection aeabi_pauthabi, 1, uleb128
// ERR: error: Expecitng optionality parameter
// ERR-NEXT: Hint: use 'optional' | 'required'
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, 1, uleb128

.aeabi_subsection aeabi_pauthabi, ,uleb128
// ERR: error: Expecitng optionality parameter
// ERR-NEXT: Hint: use 'optional' | 'required'
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, ,uleb128

.aeabi_subsection aeabi_pauthabi,uleb128
// ERR: error: unknown AArch64 build attributes optionality, expecting required|optional: uleb128
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi,uleb128

.aeabi_subsection aeabi_pauthabi uleb128
// ERR: expected comma
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi uleb128

.aeabi_subsection aeabi_pauthabi, required
// ERR: error: expected comma
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required

.aeabi_subsection aeabi_pauthabi, required,
// ERR: error: Expecitng type parameter
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required,

.aeabi_subsection aeabi_pauthabi, required, a
// ERR: error: unknown AArch64 build attributes type, expecting uleb128|ntbs: a
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required, a

.aeabi_subsection aeabi_pauthabi, required, 1
// ERR: error: Expecitng type parameter
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required, 1

.aeabi_subsection aeabi_pauthabi
// ERR: error: expected comma
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi

.aeabi_subsection aeabi_feature_and_bits, a, uleb128
// ERR: error: unknown AArch64 build attributes optionality, expecting required|optional: a
// ERR-NEXT: .aeabi_subsection aeabi_feature_and_bits, a, uleb128

.aeabi_subsection aeabi_feature_and_bits, optional, 1
// ERR: error: Expecitng type parameter
// ERR-NEXT: .aeabi_subsection aeabi_feature_and_bits, optional, 1