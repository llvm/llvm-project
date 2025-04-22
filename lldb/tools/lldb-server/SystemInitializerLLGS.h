//===-- SystemInitializerLLGS.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_SYSTEMINITIALIZERLLGS_H
#define LLDB_TOOLS_LLDB_SERVER_SYSTEMINITIALIZERLLGS_H

#include "lldb/Initialization/SystemInitializer.h"
#include "lldb/Initialization/SystemInitializerCommon.h"
#include "lldb/Utility/FileSpec.h"

class SystemInitializerLLGS : public lldb_private::SystemInitializerCommon {
public:
  SystemInitializerLLGS()
      : SystemInitializerCommon(
            // Finding the shared libraries directory on lldb-server is broken
            // since lldb-server isn't dynamically linked with liblldb.so.
            // Clearing the filespec here causes GetShlibDir to fail and
            // GetSupportExeDir to fall-back to using the binary path instead.
            [](lldb_private::FileSpec &file) { file.Clear(); }) {}

  llvm::Error Initialize() override;
  void Terminate() override;
};

#endif // LLDB_TOOLS_LLDB_SERVER_SYSTEMINITIALIZERLLGS_H
