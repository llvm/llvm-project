//===---------------- ViewerLauncher.h - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ViewerLauncher handles launching the Python web server to visualize
// the collected compilation data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADVISOR_CORE_VIEWERLAUNCHER_H
#define LLVM_ADVISOR_CORE_VIEWERLAUNCHER_H

#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace advisor {

class ViewerLauncher {
public:
  static llvm::Expected<int> launch(const std::string &outputDir,
                                    int port = 8000);

private:
  static llvm::Expected<std::string> findPythonExecutable();
  static llvm::Expected<std::string> getViewerScript();
};

} // namespace advisor
} // namespace llvm

#endif // LLVM_ADVISOR_CORE_VIEWERLAUNCHER_H
