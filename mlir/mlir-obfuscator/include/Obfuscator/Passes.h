#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

// Creates the String Encryption Pass
std::unique_ptr<Pass> createStringEncryptPass(llvm::StringRef key = "");

// Creates the Symbol Obfuscation / Renaming Pass
std::unique_ptr<Pass> createSymbolObfuscatePass(llvm::StringRef key = "");

} // namespace mlir
