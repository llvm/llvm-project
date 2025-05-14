//===-- DAPTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolBase.h"
#include "Transport.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Pipe.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using lldb_private::File;
using lldb_private::NativeFile;
using lldb_private::Pipe;

class DAPTest : public testing::Test {
protected:
  Pipe input;
  Pipe output;
  std::unique_ptr<Transport> to_dap;
  std::unique_ptr<Transport> from_dap;

  void SetUp() override {
    ASSERT_THAT_ERROR(input.CreateNew(false).ToError(), Succeeded());
    ASSERT_THAT_ERROR(output.CreateNew(false).ToError(), Succeeded());
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
};

TEST_F(DAPTest, SendProtocolMessages) {
  DAP dap{/*log=*/nullptr, /*default_repl_mode=*/ReplMode::Auto,
          /*pre_init_commands=*/{}, /*transport=*/*to_dap};
  dap.Send(Event{/*event=*/"my-event", /*body=*/std::nullopt});
  ASSERT_THAT_EXPECTED(from_dap->Read(std::chrono::milliseconds(1)),
                       HasValue(testing::VariantWith<Event>(testing::FieldsAre(
                           /*event=*/"my-event", /*body=*/std::nullopt))));
}
