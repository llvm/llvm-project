//===-- LogMessage.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LogMessage_h
#define LogMessage_h

#include <string>

class LogMessage {
public:
  virtual ~LogMessage();

  virtual bool HasActivity() const = 0;

  virtual const char *GetActivity() const = 0;

  virtual std::string GetActivityChain() const = 0;

  virtual bool HasCategory() const = 0;

  virtual const char *GetCategory() const = 0;

  virtual bool HasSubsystem() const = 0;

  virtual const char *GetSubsystem() const = 0;

  // This can be expensive, so once we ask for it, we'll cache the result.
  virtual const char *GetMessage() const = 0;

protected:
  LogMessage();
};

#endif /* LogMessage_h */
