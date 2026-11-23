//===- ForceFunctionAttrs.cpp - Force function attrs for debugging --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "IPOOptions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "forceattrs"

static bool hasConflictingFnAttr(Attribute::AttrKind Kind, Function &F) {
  switch (Kind) {
  case Attribute::AlwaysInline:
    return F.hasFnAttribute(Attribute::NoInline);

  case Attribute::NoInline:
    return F.hasFnAttribute(Attribute::AlwaysInline);

  case Attribute::OptimizeNone:
    return F.hasFnAttribute(Attribute::MinSize) ||
           F.hasFnAttribute(Attribute::OptimizeForSize) ||
           F.hasFnAttribute(Attribute::OptimizeForDebugging);

  case Attribute::MinSize:
    return F.hasFnAttribute(Attribute::OptimizeNone) ||
           F.hasFnAttribute(Attribute::OptimizeForDebugging);

  case Attribute::OptimizeForSize:
    return F.hasFnAttribute(Attribute::OptimizeNone) ||
           F.hasFnAttribute(Attribute::OptimizeForDebugging);

  case Attribute::OptimizeForDebugging:
    return F.hasFnAttribute(Attribute::OptimizeNone) ||
           F.hasFnAttribute(Attribute::MinSize) ||
           F.hasFnAttribute(Attribute::OptimizeForSize);

  default:
    return false;
  }
}

/// If F has any forced attributes given on the command line, add them.
/// If F has any forced remove attributes given on the command line, remove
/// them. When both force and force-remove are given to a function, the latter
/// takes precedence.
static void forceAttributes(const IPOOptions &Opts, Function &F) {
  auto ParseFunctionAndAttr = [&](StringRef S) {
    StringRef AttributeText;
    if (S.contains(':')) {
      auto KV = StringRef(S).split(':');
      if (KV.first != F.getName())
        return Attribute::None;
      AttributeText = KV.second;
    } else {
      AttributeText = S;
    }
    auto Kind = Attribute::getAttrKindFromName(AttributeText);
    if (Kind == Attribute::None || !Attribute::canUseAsFnAttr(Kind)) {
      LLVM_DEBUG(dbgs() << "ForcedAttribute: " << AttributeText
                        << " unknown or not a function attribute!\n");
    }
    return Kind;
  };

  for (const auto &S : Opts.force_attribute) {
    auto Kind = ParseFunctionAndAttr(S);
    if (Kind == Attribute::None || F.hasFnAttribute(Kind) ||
        hasConflictingFnAttr(Kind, F))
      continue;
    F.addFnAttr(Kind);
  }

  for (const auto &S : Opts.force_remove_attribute) {
    auto Kind = ParseFunctionAndAttr(S);
    if (Kind == Attribute::None || !F.hasFnAttribute(Kind))
      continue;
    F.removeFnAttr(Kind);
  }
}

PreservedAnalyses ForceFunctionAttrsPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  const IPOOptions &Opts = getIPOOptions(M);
  bool Changed = false;
  if (!Opts.forceattrs_csv_path.empty()) {
    auto BufferOrError = MemoryBuffer::getFileOrSTDIN(Opts.forceattrs_csv_path);
    if (!BufferOrError) {
      std::error_code EC = BufferOrError.getError();
      M.getContext().emitError("cannot open CSV file: " + EC.message());
      return PreservedAnalyses::all();
    }

    StringRef Buffer = BufferOrError.get()->getBuffer();
    auto MemoryBuffer = MemoryBuffer::getMemBuffer(Buffer);
    line_iterator It(*MemoryBuffer);
    for (; !It.is_at_end(); ++It) {
      auto SplitPair = It->split(',');
      if (SplitPair.second.empty())
        continue;
      Function *Func = M.getFunction(SplitPair.first);
      if (Func) {
        if (Func->isDeclaration())
          continue;
        auto SecondSplitPair = SplitPair.second.split('=');
        if (!SecondSplitPair.second.empty()) {
          Func->addFnAttr(SecondSplitPair.first, SecondSplitPair.second);
          Changed = true;
        } else {
          auto AttrKind = Attribute::getAttrKindFromName(SplitPair.second);
          if (AttrKind != Attribute::None &&
              Attribute::canUseAsFnAttr(AttrKind) &&
              !hasConflictingFnAttr(AttrKind, *Func)) {
            // TODO: There could be string attributes without a value, we should
            // support those, too.
            Func->addFnAttr(AttrKind);
            Changed = true;
          } else
            errs() << "Cannot add " << SplitPair.second
                   << " as an attribute name.\n";
        }
      } else {
        errs() << "Function in CSV file at line " << It.line_number()
               << " does not exist.\n";
        // TODO: `report_fatal_error at end of pass for missing functions.
        continue;
      }
    }
  }
  if (!Opts.force_attribute.empty() || !Opts.force_remove_attribute.empty()) {
    for (Function &F : M.functions())
      forceAttributes(Opts, F);
    Changed = true;
  }
  // Just conservatively invalidate analyses if we've made any changes, this
  // isn't likely to be important.
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
