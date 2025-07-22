//===-- TestBase.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestBase.h"
#include "Protocol/ProtocolBase.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Pipe.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using namespace lldb_dap_tests;
using lldb_private::File;
using lldb_private::NativeFile;
using lldb_private::Pipe;

void PipeBase::SetUp() {
  ASSERT_THAT_ERROR(input.CreateNew(false).ToError(), Succeeded());
  ASSERT_THAT_ERROR(output.CreateNew(false).ToError(), Succeeded());
}

void TransportBase::SetUp() {
  PipeBase::SetUp();
  to_dap = std::make_unique<Transport>(
      "to_dap", nullptr,
      std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                   File::eOpenOptionReadOnly,
                                   NativeFile::Unowned),
      std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                   File::eOpenOptionWriteOnly,
                                   NativeFile::Unowned));
  from_dap = std::make_unique<Transport>(
      "from_dap", nullptr,
      std::make_shared<NativeFile>(output.GetReadFileDescriptor(),
                                   File::eOpenOptionReadOnly,
                                   NativeFile::Unowned),
      std::make_shared<NativeFile>(input.GetWriteFileDescriptor(),
                                   File::eOpenOptionWriteOnly,
                                   NativeFile::Unowned));
}

void DAPTestBase::SetUp() {
  TransportBase::SetUp();
  dap = std::make_unique<DAP>(
      /*log=*/nullptr,
      /*default_repl_mode=*/ReplMode::Auto,
      /*pre_init_commands=*/std::vector<std::string>(),
      /*transport=*/*to_dap);
}

std::vector<Message> DAPTestBase::DrainOutput() {
  std::vector<Message> msgs;
  output.CloseWriteFileDescriptor();
  while (true) {
    Expected<Message> next = from_dap->Read(std::chrono::milliseconds(1));
    if (!next) {
      consumeError(next.takeError());
      break;
    }
    msgs.push_back(*next);
  }
  return msgs;
}
