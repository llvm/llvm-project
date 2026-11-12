//===-- DataBufferTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataBufferLLVM.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace lldb_private;
using namespace lldb;

TEST(DataBufferTest, RTTI) {
  {
    DataBufferSP data_buffer_sp = std::make_shared<DataBufferHeap>();
    DataBuffer *data_buffer = data_buffer_sp.get();

    EXPECT_TRUE(llvm::isa<DataBuffer>(data_buffer));
    EXPECT_TRUE(llvm::isa<WritableDataBuffer>(data_buffer));
    EXPECT_TRUE(llvm::isa<DataBufferHeap>(data_buffer));
    EXPECT_FALSE(llvm::isa<DataBufferLLVM>(data_buffer));
  }

  {
    llvm::StringRef data;
    DataBufferSP data_buffer_sp = std::make_shared<DataBufferLLVM>(
        llvm::MemoryBuffer::getMemBufferCopy(data));
    DataBuffer *data_buffer = data_buffer_sp.get();

    EXPECT_TRUE(llvm::isa<DataBuffer>(data_buffer));
    EXPECT_TRUE(llvm::isa<DataBufferLLVM>(data_buffer));
    EXPECT_FALSE(llvm::isa<WritableDataBuffer>(data_buffer));
    EXPECT_FALSE(llvm::isa<WritableDataBufferLLVM>(data_buffer));
    EXPECT_FALSE(llvm::isa<DataBufferHeap>(data_buffer));
  }

  {
    DataBufferSP data_buffer_sp = std::make_shared<WritableDataBufferLLVM>(
        llvm::WritableMemoryBuffer::getNewMemBuffer(1));
    DataBuffer *data_buffer = data_buffer_sp.get();

    EXPECT_TRUE(llvm::isa<DataBuffer>(data_buffer));
    EXPECT_TRUE(llvm::isa<WritableDataBuffer>(data_buffer));
    EXPECT_TRUE(llvm::isa<WritableDataBufferLLVM>(data_buffer));
    EXPECT_FALSE(llvm::isa<DataBufferLLVM>(data_buffer));
    EXPECT_FALSE(llvm::isa<DataBufferHeap>(data_buffer));
  }
}
