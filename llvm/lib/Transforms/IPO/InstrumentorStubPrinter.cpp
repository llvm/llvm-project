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
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>
#include <system_error>

namespace llvm {
namespace instrumentor {

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

static raw_fd_ostream *createOutputStream(StringRef Name) {
  std::error_code EC;
  auto *Out = new raw_fd_ostream(Name, EC);
  if (EC) {
    errs() << "WARNING: Failed to open instrumentor stub runtime file for "
              "writing: "
           << EC.message() << "\n";
    delete Out;
    Out = nullptr;
  } else {
    *Out << "// LLVM Instrumentor stub runtime\n\n";
    *Out << "#include <stdint.h>\n";
    *Out << "#include <stdio.h>\n\n";
  }

  return Out;
}

void printRuntimeStub(const InstrumentationConfig &IConf,
                      StringRef StubRuntimeName, const Module &M) {
  if (StubRuntimeName.empty())
    return;

  auto *Out = createOutputStream(StubRuntimeName);
  if (!Out)
    return;

  for (auto &ChoiceMap : IConf.IChoices) {
    for (auto &[_, IO] : ChoiceMap) {
      if (!IO->Enabled)
        continue;
      IRTCallDescription IRTCallDesc(*IO, IO->getRetTy(M.getContext()));
      const auto Signatures = IRTCallDesc.createCSignature(IConf);
      const auto Bodies = IRTCallDesc.createCBodies();
      if (!Signatures.first.empty()) {
        *Out << Signatures.first << " {\n";
        *Out << "  " << Bodies.first << "}\n\n";
      }
      if (!Signatures.second.empty()) {
        *Out << Signatures.second << " {\n";
        *Out << "  " << Bodies.second << "}\n\n";
      }
    }
  }

  delete Out;
}

} // end namespace instrumentor
} // end namespace llvm
