//===-- SystemInitializerLLGS.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemInitializerLLGS.h"

#if defined(__APPLE__)
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
using HostObjectFile = ObjectFileMachO;
#elif defined(_WIN32)
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
using HostObjectFile = ObjectFilePECOFF;
#else
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
using HostObjectFile = ObjectFileELF;
#endif

using namespace lldb_private;

llvm::Error SystemInitializerLLGS::Initialize() {
  if (auto e = SystemInitializerCommon::Initialize())
    return e;

  HostObjectFile::Initialize();

  return llvm::Error::success();
}

void SystemInitializerLLGS::Terminate() {
  HostObjectFile::Terminate();
  SystemInitializerCommon::Terminate();
}
