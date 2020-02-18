//===-- ActivityStore.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_ACTIVITYSTORE_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_ACTIVITYSTORE_H

#include <string>

#include "ActivityStreamSPI.h"

class ActivityStore {
public:
  virtual ~ActivityStore();

  virtual const char *GetActivityForID(os_activity_id_t activity_id) const = 0;

  virtual std::string
  GetActivityChainForID(os_activity_id_t activity_id) const = 0;

protected:
  ActivityStore();
};

#endif // LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_ACTIVITYSTORE_H
