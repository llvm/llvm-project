//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ENCRYPTION_COLOR_H
#define LLVM_IR_ENCRYPTION_COLOR_H

#include "llvm/IR/Metadata.h"

enum EncryptionColor {
    /// The value is unencrypted
    Plaintext,

    /// The value is encrypted
    Encrypted,

    /// A pointer to a plaintext buffer. These always have a plaintext index.
    PlainPtr,

    /// A pointer to an encrypted buffer with a plaintext index.
    EncryptedPtrEncryptedPlainIndex,

    /// A pointer to an encrypted buffer with an encrypted index
    EncryptedPtrEncryptedIndex,
    Last = EncryptedPtrEncryptedIndex
};

#endif
