//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Option/LibraryOptions.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::opt;

LibraryOptTable::~LibraryOptTable() = default;

void *llvm::opt::allocateListStorage(size_t Size, size_t Alignment) {
  // Never destroyed: list members refer to it until exit.
  static BumpPtrAllocator &Alloc = *new BumpPtrAllocator;
  return Alloc.Allocate(Size, Align(Alignment));
}

void LibraryOptionsParser::forEachOption(
    function_ref<void(StringRef, StringRef, StringRef)> Fn) const {
  const OptTable &T = Table();
  for (unsigned ID = 1, E = T.getNumOptions(); ID <= E; ++ID) {
    unsigned Kind = T.getOptionKind(ID);
    if (Kind != Option::FlagOrEqClass && Kind != Option::SeparateOrEqClass)
      continue;
    StringRef V = T.getOptionMetaVar(ID);
    std::string MetaVar;
    if (Kind == Option::SeparateOrEqClass)
      MetaVar = ("=" + (V.empty() ? StringRef("<value>") : V)).str();
    else if (!V.empty())
      MetaVar = ("[=" + V + "]").str();
    Fn(T.getOptionName(ID), MetaVar, T.getOptionHelpText(ID));
  }
}

Error LibraryOptionsParser::parse(ArrayRef<const char *> Args,
                                  unsigned &Consumed) {
  InputArgList List(Args.begin(), Args.end());
  Consumed = 0;
  std::unique_ptr<Arg> A = Table().ParseOneArg(List, Consumed);
  Consumed = std::min<unsigned>(Consumed, Args.size());
  if (!A)
    return createStringError("option '" + Twine(Args[0]) +
                             "' requires an argument");
  if (A->getOption().getKind() == Option::UnknownClass)
    return createStringError("unknown argument '" + Twine(Args[0]) + "'");
  if (!Apply(*A))
    return createStringError("invalid value '" + Twine(A->getValue()) +
                             "' in '" + A->getAsString(List) + "'");
  return Error::success();
}
