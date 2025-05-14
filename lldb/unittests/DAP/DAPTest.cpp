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
  std::unique_ptr<Transport> toDAP;
  std::unique_ptr<Transport> fromDAP;

  void SetUp() override {
    ASSERT_THAT_ERROR(input.CreateNew(false).ToError(), Succeeded());
    ASSERT_THAT_ERROR(output.CreateNew(false).ToError(), Succeeded());
    toDAP = std::make_unique<Transport>(
        "toDAP", nullptr,
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));
    fromDAP = std::make_unique<Transport>(
        "fromDAP", nullptr,
        std::make_shared<NativeFile>(output.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(input.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));
  }
};

TEST_F(DAPTest, SendProtocolMessages) {
  DAP dap{nullptr, ReplMode::Auto, {}, *toDAP};
  dap.Send(Event{"my-event", std::nullopt});
  ASSERT_THAT_EXPECTED(fromDAP->Read(std::chrono::milliseconds(1)),
                       HasValue(testing::VariantWith<Event>(testing::FieldsAre(
                           /*event=*/"my-event", /*body=*/std::nullopt))));
}
