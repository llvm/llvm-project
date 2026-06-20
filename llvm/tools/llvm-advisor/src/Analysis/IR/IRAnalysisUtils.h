//===--- IRAnalysisUtils.h - LLVM Advisor --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"
#include "llvm/IR/Module.h"

namespace llvm::advisor {

/// Parse an LLVM IR file into a Module.  The caller must keep Ctx alive
/// for the lifetime of the returned Module.
Expected<std::unique_ptr<Module>> parseIRModule(StringRef Path,
                                                 LLVMContext &Ctx);

/// Parse a bitcode file into a Module.  The caller must keep Ctx alive
/// for the lifetime of the returned Module.
Expected<std::unique_ptr<Module>> parseBitcodeModule(StringRef Path,
                                                        LLVMContext &Ctx);

/// Run Analyze over the textual IR module described by Context.  If the
/// IR artifact is missing, returns an unavailable result.  Parse errors are
/// forwarded as-is.  The temporary LLVMContext and Module are passed
/// to Analyze, which must extract all required data before returning.
template <typename Func>
Expected<std::unique_ptr<CapabilityResult>>
withIRModule(const CapabilityContext &Context, StringRef CapabilityID,
             StringRef UnitID, Func &&Analyze) {
  if (Context.IRPath.empty())
    return makeUnavailableResult(CapabilityID, UnitID, "missing IR artifact");
  LLVMContext Ctx;
  auto MOrErr = parseIRModule(Context.IRPath, Ctx);
  if (!MOrErr)
    return MOrErr.takeError();
  return std::forward<Func>(Analyze)(Ctx, **MOrErr);
}

/// Same as withIRModule, but parses bitcode instead of textual IR.
template <typename Func>
Expected<std::unique_ptr<CapabilityResult>>
withBitcodeModule(const CapabilityContext &Context, StringRef CapabilityID,
                  StringRef UnitID, Func &&Analyze) {
  if (Context.IRPath.empty())
    return makeUnavailableResult(CapabilityID, UnitID, "missing IR artifact");
  LLVMContext Ctx;
  auto MOrErr = parseBitcodeModule(Context.IRPath, Ctx);
  if (!MOrErr)
    return MOrErr.takeError();
  return std::forward<Func>(Analyze)(Ctx, **MOrErr);
}

} // namespace llvm::advisor
