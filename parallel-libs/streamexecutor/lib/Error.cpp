//===-- Error.cpp - Error handling ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Types for returning recoverable errors.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/Error.h"

#include "llvm/ADT/StringRef.h"

namespace {

// An error with a string message describing the cause.
class StreamExecutorError : public llvm::ErrorInfo<StreamExecutorError> {
public:
  StreamExecutorError(llvm::StringRef Message) : Message(Message.str()) {}

  void log(llvm::raw_ostream &OS) const override { OS << Message; }

  std::error_code convertToErrorCode() const override {
    llvm_unreachable(
        "StreamExecutorError does not support conversion to std::error_code");
  }

  std::string getErrorMessage() const { return Message; }

  static char ID;

private:
  std::string Message;
};

char StreamExecutorError::ID = 0;

} // namespace

namespace streamexecutor {

Error make_error(const Twine &Message) {
  return llvm::make_error<StreamExecutorError>(Message.str());
}

std::string consumeAndGetMessage(Error &&E) {
  if (!E)
    return "success";
  std::string Message;
  llvm::handleAllErrors(std::move(E),
                        [&Message](const StreamExecutorError &SEE) {
                          Message = SEE.getErrorMessage();
                        });
  return Message;
}

void dieIfError(Error &&E) {
  if (E) {
    std::fprintf(stderr, "Error encountered: %s.\n",
                 streamexecutor::consumeAndGetMessage(std::move(E)).c_str());
    std::exit(EXIT_FAILURE);
  }
}

} // namespace streamexecutor
