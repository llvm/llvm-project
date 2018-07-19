//===- llvm/Testing/Support/Error.cpp -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Testing/Support/Error.h"

#include "llvm/ADT/StringRef.h"

using namespace llvm;

llvm::detail::ErrorHolder llvm::detail::TakeError(llvm::Error Err) {
  bool Succeeded = !static_cast<bool>(Err);
  std::string Message;
  if (!Succeeded)
    Message = toString(std::move(Err));
  return {Succeeded, Message};
}
