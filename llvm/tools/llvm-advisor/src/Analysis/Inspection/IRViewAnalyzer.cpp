//===------------------- IRViewAnalyzer.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/IRViewAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
IRViewAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    std::string Text;
    raw_string_ostream OS(Text);
    M.print(OS, nullptr);
    OS.flush();

    return makeJSONResult(CapID, UnitID, json::Object{
        {"module", M.getModuleIdentifier()},
        {"ir", Text},
    });
  });
}
