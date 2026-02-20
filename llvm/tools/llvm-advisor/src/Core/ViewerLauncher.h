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

#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_VIEWERLAUNCHER_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_VIEWERLAUNCHER_H

#include "llvm/Support/Error.h"
#include <string>


namespace llvm::advisor {

class ViewerLauncher {
public:
  static auto launch(const std::string &outputDir,
                                    int port = 8000) -> llvm::Expected<int>;

private:
  static auto findPythonExecutable() -> llvm::Expected<std::string>;
  static auto getViewerScript() -> llvm::Expected<std::string>;
};

} // namespace llvm::advisor


#endif // LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_VIEWERLAUNCHER_H
