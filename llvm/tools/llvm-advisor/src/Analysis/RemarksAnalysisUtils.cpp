//===--- RemarksAnalysisUtils.cpp - LLVM Advisor -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/RemarksAnalysisUtils.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::advisor;

Error llvm::advisor::foreachRemark(StringRef Path, RemarkVisitor Visitor) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createStringError(MB.getError(),
                             Twine("cannot read remarks: ") + Path);

  Expected<std::unique_ptr<remarks::RemarkParser>> Parser =
      remarks::createRemarkParser(remarks::Format::Auto,
                                   MB.get()->getBuffer());
  if (!Parser)
    return Parser.takeError();

  while (true) {
    Expected<std::unique_ptr<remarks::Remark>> Next = (*Parser)->next();
    if (!Next) {
      Error E = Next.takeError();
      if (E.isA<remarks::EndOfFileError>()) {
        consumeError(std::move(E));
        break;
      }
      return E;
    }
    if (Error E = Visitor(**Next))
      return E;
  }
  return Error::success();
}
