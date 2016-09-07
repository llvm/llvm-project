//===-- RNBServices.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Christopher Friesen on 3/21/08.
//
//===----------------------------------------------------------------------===//

#ifndef __RNBServices_h__
#define __RNBServices_h__

#include "RNBDefs.h"
#include <string>

#define DTSERVICES_APP_FRONTMOST_KEY CFSTR("isFrontApp")
#define DTSERVICES_APP_PATH_KEY CFSTR("executablePath")
#define DTSERVICES_APP_ICON_PATH_KEY CFSTR("iconPath")
#define DTSERVICES_APP_DISPLAY_NAME_KEY CFSTR("displayName")
#define DTSERVICES_APP_PID_KEY CFSTR("pid")

int ListApplications(std::string &plist, bool opt_runningApps,
                     bool opt_debuggable);

#endif // __RNBServices_h__
