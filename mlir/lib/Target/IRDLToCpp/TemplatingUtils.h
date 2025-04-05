//===- TemplatingUtils.h - Templater for text templates -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IRDLTOCPP_TEMPLATE_UTILS_H
#define IRDLTOCPP_TEMPLATE_UTILS_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <variant>

namespace mlir::irdl::detail {

using dictionary = llvm::StringMap<llvm::SmallString<8>>;

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

  void render(llvm::raw_ostream &out, const dictionary &replacements) const {
    for (auto instruction : bytecode) {
      std::visit(
          [&](auto &&inst) {
            using T = std::decay_t<decltype(inst)>;
            if constexpr (std::is_same_v<T, LiteralToken>) {
              out << inst.text;
            } else if constexpr (std::is_same_v<T, ReplacementToken>) {
              auto replacement = replacements.find(inst.keyName);
#ifndef NDEBUG
              if (replacement == replacements.end()) {
                llvm::errs()
                    << "Missing template key: " << inst.keyName << "\n";
                llvm_unreachable("Missing template key");
              }
#endif
              out << replacement->second;
            } else {
              static_assert(false, "non-exhaustive visitor!");
            }
          },
          instruction);
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

#endif // #ifndef IRDLTOCPP_TEMPLATE_UTILS_H
