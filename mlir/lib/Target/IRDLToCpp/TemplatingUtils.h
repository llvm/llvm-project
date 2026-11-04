//===- TemplatingUtils.h - Templater for text templates -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_IRDLTOCPP_TEMPLATINGUTILS_H
#define MLIR_LIB_TARGET_IRDLTOCPP_TEMPLATINGUTILS_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <variant>
#include <vector>

namespace mlir::irdl::detail {

/// A dictionary stores a mapping of template variable names to their assigned
/// string values.
using dictionary = llvm::StringMap<llvm::SmallString<8>>;

/// Template Code as used by IRDL-to-Cpp.
///
/// For efficiency, produces a bytecode representation of an input template.
///   - LiteralToken: A contiguous stream of characters to be printed
///   - ReplacementToken: A template variable that will be replaced
class Template {
public:
  Template(llvm::StringRef str) {
    bool processingReplacementToken = false;
    while (!str.empty()) {
      auto [token, remainder] = str.split("__");

      if (processingReplacementToken) {
        assert(!token.empty() && "replacement name cannot be empty");
        bytecode.emplace_back(ReplacementToken{token});
      } else {
        if (!token.empty())
          bytecode.emplace_back(LiteralToken{token});
      }

      processingReplacementToken = !processingReplacementToken;
      str = remainder;
    }
  }

  /// Render will apply a dictionary to the Template and send the rendered
  /// result to the specified output stream.
  void render(llvm::raw_ostream &out, const dictionary &replacements) const {
    for (auto instruction : bytecode) {
      if (auto *inst = std::get_if<LiteralToken>(&instruction)) {
        out << inst->text;
        continue;
      }

      if (auto *inst = std::get_if<ReplacementToken>(&instruction)) {
        auto replacement = replacements.find(inst->keyName);
#ifndef NDEBUG
        if (replacement == replacements.end()) {
          llvm::errs() << "Missing template key: " << inst->keyName << "\n";
          llvm_unreachable("Missing template key");
        }
#endif
        out << replacement->second;
        continue;
      }

      llvm_unreachable("non-exhaustive bytecode visit");
    }
  }

private:
  struct LiteralToken {
    llvm::StringRef text;
  };

  struct ReplacementToken {
    llvm::StringRef keyName;
  };

  std::vector<std::variant<LiteralToken, ReplacementToken>> bytecode;
};

} // namespace mlir::irdl::detail

#endif // MLIR_LIB_TARGET_IRDLTOCPP_TEMPLATINGUTILS_H
