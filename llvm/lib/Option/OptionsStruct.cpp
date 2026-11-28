//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Option/OptionsStruct.h"
#include "llvm/Option/ArgList.h"

using namespace llvm;
using namespace llvm::opt;

OptionsStructTable::~OptionsStructTable() = default;

Error opt::detail::parseToolArgsImpl(const OptTable &T, int Argc,
                                     const char *const *Argv,
                                     StringSaver &Saver,
                                     function_ref<bool(const Arg &)> Apply,
                                     SmallVectorImpl<StringRef> &Inputs) {
  unsigned Unknown = 1;
  while (T.getOptionKind(Unknown) != Option::UnknownClass)
    ++Unknown;
  std::string Err;
  InputArgList List = T.parseArgs(Argc, const_cast<char *const *>(Argv),
                                  Unknown, Saver, [&](StringRef Msg) {
                                    if (Err.empty())
                                      Err = Msg;
                                  });
  if (!Err.empty())
    return createStringError(Err);
  for (const Arg *A : List) {
    if (A->getOption().getKind() == Option::InputClass)
      Inputs.push_back(A->getValue());
    else if (!Apply(*A))
      return createStringError("invalid value '" + Twine(A->getValue()) +
                               "' in '" + A->getAsString(List) + "'");
  }
  return Error::success();
}
