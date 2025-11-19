//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of custom command-line argument parsers
/// using llvm::cl.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_COMMANDLINE_HPP
#define MATHTEST_COMMANDLINE_HPP

#include "mathtest/TestConfig.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include <string>

namespace llvm {
namespace cl {

struct TestConfigsArg {
  enum class Mode { Default, All, Explicit } Mode = Mode::Default;
  llvm::SmallVector<mathtest::TestConfig, 4> Explicit;
};

template <> class parser<TestConfigsArg> : public basic_parser<TestConfigsArg> {
public:
  parser(Option &O) : basic_parser<TestConfigsArg>(O) {}

  static bool isAllowed(const mathtest::TestConfig &Config) {
    static const llvm::SmallVector<mathtest::TestConfig, 4> &AllTestConfigs =
        mathtest::getAllTestConfigs();

    return llvm::is_contained(AllTestConfigs, Config);
  }

  bool parse(Option &O, StringRef ArgName, StringRef ArgValue,
             TestConfigsArg &Val) {
    ArgValue = ArgValue.trim();
    if (ArgValue.empty())
      return O.error(
          "Expected '" + getValueName() +
          "', but got an empty string. Omit the flag to use defaults");

    if (ArgValue.equals_insensitive("all")) {
      Val.Mode = TestConfigsArg::Mode::All;
      return false;
    }

    llvm::SmallVector<StringRef, 8> Pairs;
    ArgValue.split(Pairs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

    Val.Mode = TestConfigsArg::Mode::Explicit;
    Val.Explicit.clear();

    for (StringRef Pair : Pairs) {
      llvm::SmallVector<StringRef, 2> Parts;
      Pair.split(Parts, ':');

      if (Parts.size() != 2)
        return O.error("Expected '<provider>:<platform>', got '" + Pair + "'");

      StringRef Provider = Parts[0].trim();
      StringRef Platform = Parts[1].trim();

      if (Provider.empty() || Platform.empty())
        return O.error("Provider and platform must not be empty in '" + Pair +
                       "'");

      mathtest::TestConfig Config = {Provider.str(), Platform.str()};
      if (!isAllowed(Config))
        return O.error("Invalid pair '" + Pair + "'");

      Val.Explicit.push_back(Config);
    }

    return false;
  }

  StringRef getValueName() const override {
    return "all|provider:platform[,provider:platform...]";
  }

  void printOptionDiff(const Option &O, const TestConfigsArg &V, OptVal Default,
                       size_t GlobalWidth) const {
    printOptionNoValue(O, GlobalWidth);
  }
};
} // namespace cl
} // namespace llvm

#endif // MATHTEST_COMMANDLINE_HPP
