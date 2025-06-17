//===-- InstrumentorStubPrinter.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>
#include <system_error>

namespace llvm {
namespace instrumentor {

/// Get the string representation of an argument with type \p Ty. Two strings
/// are returned: one for direct arguments and another for indirect arguments.
/// The flags in \p Flags describe the properties of the argument. See
/// IRTArg::IRArgFlagTy.
static std::pair<std::string, std::string> getAsCType(Type *Ty,
                                                      unsigned Flags) {
  if (Ty->isIntegerTy()) {
    auto BW = Ty->getIntegerBitWidth();
    if (BW == 1)
      return {"bool ", "bool *"};
    auto S = "int" + std::to_string(BW) + "_t ";
    return {S, S + "*"};
  }
  if (Ty->isPointerTy())
    return {Flags & IRTArg::STRING ? "char *" : "void *", "void **"};
  if (Ty->isFloatTy())
    return {"float ", "float *"};
  if (Ty->isDoubleTy())
    return {"double ", "double *"};
  return {"<>", "<>"};
}

/// Get the string representation of the C printf format of an argument with
/// type \p Ty. The flags in \p Flags describe the properties of the argument.
/// See IRTArg::IRArgFlagTy.
static std::string getPrintfFormatString(Type *Ty, unsigned Flags) {
  if (Ty->isIntegerTy()) {
    if (Ty->getIntegerBitWidth() > 32) {
      assert(Ty->getIntegerBitWidth() == 64);
      return "%lli";
    }
    return "%i";
  }
  if (Ty->isPointerTy())
    return Flags & IRTArg::STRING ? "%s" : "%p";
  if (Ty->isFloatTy())
    return "%f";
  if (Ty->isDoubleTy())
    return "%lf";
  return "<>";
}

std::pair<std::string, std::string> IRTCallDescription::createCBodies() const {
  std::string DirectFormat = "printf(\"" + IO.getName().str() +
                             (IO.IP.isPRE() ? " pre" : " post") + " -- ";
  std::string IndirectFormat = DirectFormat;
  std::string DirectArg, IndirectArg, DirectReturnValue, IndirectReturnValue;

  auto AddToFormats = [&](Twine S) {
    DirectFormat += S.str();
    IndirectFormat += S.str();
  };
  auto AddToArgs = [&](Twine S) {
    DirectArg += S.str();
    IndirectArg += S.str();
  };
  bool First = true;
  for (auto &IRArg : IO.IRTArgs) {
    if (!IRArg.Enabled)
      continue;
    if (!First)
      AddToFormats(", ");
    First = false;
    AddToArgs(", " + IRArg.Name);
    AddToFormats(IRArg.Name + ": ");
    if (NumReplaceableArgs == 1 && (IRArg.Flags & IRTArg::REPLACABLE)) {
      DirectReturnValue = IRArg.Name;
      if (!isPotentiallyIndirect(IRArg))
        IndirectReturnValue = IRArg.Name;
    }
    if (!isPotentiallyIndirect(IRArg)) {
      AddToFormats(getPrintfFormatString(IRArg.Ty, IRArg.Flags));
    } else {
      DirectFormat += getPrintfFormatString(IRArg.Ty, IRArg.Flags);
      IndirectFormat += "%p";
      IndirectArg += "_ptr";
      // Add the indirect argument size
      if (!(IRArg.Flags & IRTArg::INDIRECT_HAS_SIZE)) {
        IndirectFormat += ", " + IRArg.Name.str() + "_size: %i";
        IndirectArg += ", " + IRArg.Name.str() + "_size";
      }
    }
  }

  std::string DirectBody = DirectFormat + "\\n\"" + DirectArg + ");\n";
  std::string IndirectBody = IndirectFormat + "\\n\"" + IndirectArg + ");\n";
  if (RetTy)
    IndirectReturnValue = DirectReturnValue = "0";
  if (!DirectReturnValue.empty())
    DirectBody += "  return " + DirectReturnValue + ";\n";
  if (!IndirectReturnValue.empty())
    IndirectBody += "  return " + IndirectReturnValue + ";\n";
  return {DirectBody, IndirectBody};
}

std::pair<std::string, std::string>
IRTCallDescription::createCSignature(const InstrumentationConfig &IConf) const {
  SmallVector<std::string> DirectArgs, IndirectArgs;
  std::string DirectRetTy = "void ", IndirectRetTy = "void ";
  for (auto &IRArg : IO.IRTArgs) {
    if (!IRArg.Enabled)
      continue;
    const auto &[DirectArgTy, IndirectArgTy] =
        getAsCType(IRArg.Ty, IRArg.Flags);
    std::string DirectArg = DirectArgTy + IRArg.Name.str();
    std::string IndirectArg = IndirectArgTy + IRArg.Name.str() + "_ptr";
    std::string IndirectArgSize = "int32_t " + IRArg.Name.str() + "_size";
    DirectArgs.push_back(DirectArg);
    if (NumReplaceableArgs == 1 && (IRArg.Flags & IRTArg::REPLACABLE)) {
      DirectRetTy = DirectArgTy;
      if (!isPotentiallyIndirect(IRArg))
        IndirectRetTy = DirectArgTy;
    }
    if (!isPotentiallyIndirect(IRArg)) {
      IndirectArgs.push_back(DirectArg);
    } else {
      IndirectArgs.push_back(IndirectArg);
      if (!(IRArg.Flags & IRTArg::INDIRECT_HAS_SIZE))
        IndirectArgs.push_back(IndirectArgSize);
    }
  }

  auto DirectName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(), "");
  auto IndirectName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(), "_ind");
  auto MakeSignature = [&](std::string &RetTy, std::string &Name,
                           SmallVectorImpl<std::string> &Args) {
    return RetTy + Name + "(" + join(Args, ", ") + ")";
  };

  if (RetTy) {
    auto UserRetTy = getAsCType(RetTy, 0).first;
    assert((DirectRetTy == UserRetTy || DirectRetTy == "void ") &&
           (IndirectRetTy == UserRetTy || IndirectRetTy == "void ") &&
           "Explicit return type but also implicit one!");
    IndirectRetTy = DirectRetTy = UserRetTy;
  }
  if (RequiresIndirection)
    return {"", MakeSignature(IndirectRetTy, IndirectName, IndirectArgs)};
  if (!MightRequireIndirection)
    return {MakeSignature(DirectRetTy, DirectName, DirectArgs), ""};
  return {MakeSignature(DirectRetTy, DirectName, DirectArgs),
          MakeSignature(IndirectRetTy, IndirectName, IndirectArgs)};
}

void printRuntimeStub(const InstrumentationConfig &IConf,
                      StringRef StubRuntimeName, LLVMContext &Ctx) {
  if (StubRuntimeName.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(StubRuntimeName, EC);
  if (EC) {
    Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("Failed to open instrumentor stub runtime file for writing: ") +
            EC.message(),
        DS_Warning));
    return;
  }

  OS << "// LLVM Instrumentor stub runtime\n\n";
  OS << "#include <stdint.h>\n";
  OS << "#include <stdio.h>\n\n";

  for (auto &ChoiceMap : IConf.IChoices) {
    for (auto &[_, IO] : ChoiceMap) {
      if (!IO->Enabled)
        continue;
      IRTCallDescription IRTCallDesc(*IO, IO->getRetTy(Ctx));
      const auto Signatures = IRTCallDesc.createCSignature(IConf);
      const auto Bodies = IRTCallDesc.createCBodies();
      if (!Signatures.first.empty()) {
        OS << Signatures.first << " {\n";
        OS << "  " << Bodies.first << "}\n\n";
      }
      if (!Signatures.second.empty()) {
        OS << Signatures.second << " {\n";
        OS << "  " << Bodies.second << "}\n\n";
      }
    }
  }
}

} // end namespace instrumentor
} // end namespace llvm
