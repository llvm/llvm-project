//===-- MICmnLLDBBroadcaster.h ----------------------------------*- C++ -*-===//
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
#include "lldb/API/SBBroadcaster.h"

//++
//============================================================================
// Details: MI derived class from LLDB SBBroadcaster API.
//
//          *** This class (files) is a place holder until we know we need it or
//          *** not
//
//          A singleton class.
//--
class CMICmnLLDBBroadcaster : public CMICmnBase,
                              public lldb::SBBroadcaster,
                              public MI::ISingleton<CMICmnLLDBBroadcaster> {
  friend MI::ISingleton<CMICmnLLDBBroadcaster>;

  // Methods:
public:
  bool Initialize() override;
  bool Shutdown() override;
  // Methods:
private:
  /* ctor */ CMICmnLLDBBroadcaster();
  /* ctor */ CMICmnLLDBBroadcaster(const CMICmnLLDBBroadcaster &);
  void operator=(const CMICmnLLDBBroadcaster &);

  // Overridden:
private:
  // From CMICmnBase
  /* dtor */ ~CMICmnLLDBBroadcaster() override;
};
