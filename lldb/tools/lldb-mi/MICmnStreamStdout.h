//===-- MICmnStreamStdout.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MICmnBase.h"
#include "MIUtilSingletonBase.h"
#include "MIUtilString.h"
#include "MIUtilThreadBaseStd.h"

//++
//============================================================================
// Details: MI common code class. The MI driver requires this object.
//          CMICmnStreamStdout sets up and tears downs stdout for the driver.
//
//          Singleton class.
//--
class CMICmnStreamStdout : public CMICmnBase,
                           public MI::ISingleton<CMICmnStreamStdout> {
  friend class MI::ISingleton<CMICmnStreamStdout>;

  // Statics:
public:
  static bool TextToStdout(const CMIUtilString &vrTxt);
  static bool WritePrompt();

  // Methods:
public:
  bool Initialize() override;
  bool Shutdown() override;
  //
  bool Lock();
  bool Unlock();
  bool Write(const CMIUtilString &vText, const bool vbSendToLog = true);
  bool WriteMIResponse(const CMIUtilString &vText,
                       const bool vbSendToLog = true);

  // Methods:
private:
  /* ctor */ CMICmnStreamStdout();
  /* ctor */ CMICmnStreamStdout(const CMICmnStreamStdout &);
  void operator=(const CMICmnStreamStdout &);
  //
  bool WritePriv(const CMIUtilString &vText,
                 const CMIUtilString &vTxtForLogFile,
                 const bool vbSendToLog = true);

  // Overridden:
private:
  // From CMICmnBase
  /* dtor */ ~CMICmnStreamStdout() override;

  // Attributes:
private:
  CMIUtilThreadMutex m_mutex; // Mutex object for sync during writing to stream
};
