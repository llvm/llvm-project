//===-- FifoFilesTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FifoFiles.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <thread>

using namespace lldb_dap;
using namespace llvm;

namespace {

std::string MakeTempFifoPath() {
  llvm::SmallString<128> temp_path;
  llvm::sys::fs::createUniquePath("lldb-dap-fifo-%%%%%%", temp_path,
                                  /*MakeAbsolute=*/true);
  return temp_path.str().str();
}

} // namespace

TEST(FifoFilesTest, CreateAndDestroyFifoFile) {
  std::string fifo_path = MakeTempFifoPath();
  auto fifo = CreateFifoFile(fifo_path);
  EXPECT_THAT_EXPECTED(fifo, llvm::Succeeded());

  // File should exist.
  EXPECT_TRUE(llvm::sys::fs::exists(fifo_path));

  // Destructor should remove the file.
  fifo->reset();
  EXPECT_FALSE(llvm::sys::fs::exists(fifo_path));
}

TEST(FifoFilesTest, SendAndReceiveJSON) {
  std::string fifo_path = MakeTempFifoPath();
  auto fifo = CreateFifoFile(fifo_path);
  EXPECT_THAT_EXPECTED(fifo, llvm::Succeeded());

  FifoFileIO writer(fifo_path, "writer");
  FifoFileIO reader(fifo_path, "reader");

  llvm::json::Object obj;
  obj["foo"] = "bar";
  obj["num"] = 42;

  // Writer thread.
  std::thread writer_thread([&]() {
    EXPECT_THAT_ERROR(writer.SendJSON(llvm::json::Value(std::move(obj)),
                                      std::chrono::milliseconds(500)),
                      llvm::Succeeded());
  });

  // Reader thread.
  std::thread reader_thread([&]() {
    auto result = reader.ReadJSON(std::chrono::milliseconds(500));
    EXPECT_THAT_EXPECTED(result, llvm::Succeeded());
    auto *read_obj = result->getAsObject();

    ASSERT_NE(read_obj, nullptr);
    EXPECT_EQ((*read_obj)["foo"].getAsString(), "bar");
    EXPECT_EQ((*read_obj)["num"].getAsInteger(), 42);
  });

  writer_thread.join();
  reader_thread.join();
}

TEST(FifoFilesTest, ReadTimeout) {
  std::string fifo_path = MakeTempFifoPath();
  auto fifo = CreateFifoFile(fifo_path);
  EXPECT_THAT_EXPECTED(fifo, llvm::Succeeded());

  FifoFileIO reader(fifo_path, "reader");

  // No writer, should timeout.
  auto result = reader.ReadJSON(std::chrono::milliseconds(100));
  EXPECT_THAT_EXPECTED(result, llvm::Failed());
}

TEST(FifoFilesTest, WriteTimeout) {
  std::string fifo_path = MakeTempFifoPath();
  auto fifo = CreateFifoFile(fifo_path);
  EXPECT_THAT_EXPECTED(fifo, llvm::Succeeded());

  FifoFileIO writer(fifo_path, "writer");

  // No reader, should timeout.
  llvm::json::Object obj;
  obj["foo"] = "bar";
  EXPECT_THAT_ERROR(writer.SendJSON(llvm::json::Value(std::move(obj)),
                                    std::chrono::milliseconds(100)),
                    llvm::Failed());
}
