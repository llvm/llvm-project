// RUN: not llvm-mc -triple=aarch64 %s 2>&1 | FileCheck --check-prefix=ERR %s

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

.aeabi_subsection 1, required, uleb128
// ERR: error: subsection name not found
// ERR-NEXT: .aeabi_subsection 1, required, uleb128

.aeabi_subsection , required, uleb128
// ERR: error: subsection name not found
// ERR-NEXT: .aeabi_subsection , required, uleb128

.aeabi_subsection aeabi_pauthabi, a, uleb128
// ERR: error: unknown AArch64 build attributes optionality, expected required|optional: a
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, a, uleb128

.aeabi_subsection aeabi_pauthabi, a, uleb128
// ERR: error: unknown AArch64 build attributes optionality, expected required|optional: a
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, a, uleb128

.aeabi_subsection aeabi_pauthabi, 1, uleb128
// ERR: error: optionality parameter not found, expected required|optional
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, 1, uleb128

.aeabi_subsection aeabi_pauthabi, ,uleb128
// ERR: error: optionality parameter not found, expected required|optional
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, ,uleb128

.aeabi_subsection aeabi_pauthabi,uleb128
// ERR: error: unknown AArch64 build attributes optionality, expected required|optional: uleb128
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi,uleb128

.aeabi_subsection aeabi_pauthabi uleb128
// ERR: expected comma
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi uleb128

.aeabi_subsection aeabi_pauthabi, required
// ERR: error: expected comma
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required

.aeabi_subsection aeabi_pauthabi, required,
// ERR: error: type parameter not found, expected uleb128|ntbs
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required,

.aeabi_subsection aeabi_pauthabi, required, a
// ERR: error: unknown AArch64 build attributes type, expected uleb128|ntbs: a
// ERR-NEXT: .aeabi_subsection aeabi_pauthabi, required, a
