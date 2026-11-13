//===- RemarkStreamerTest.cpp - Remark streamer tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Remark/RemarkStreamer.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <atomic>
#include <string>
#include <thread>
#include <vector>

using namespace mlir;

#if LLVM_ENABLE_THREADS
TEST(RemarkStreamer, ConcurrentWrites) {
  for (auto format :
       {llvm::remarks::Format::YAML, llvm::remarks::Format::Bitstream}) {
    SCOPED_TRACE(format == llvm::remarks::Format::YAML ? "YAML" : "Bitstream");
    SmallString<128> path;
    ASSERT_FALSE(
        llvm::sys::fs::createTemporaryFile("mlir-remarks", "tmp", path));
    llvm::FileRemover cleanup(path);
    auto streamerOr =
        remark::detail::LLVMRemarkStreamer::createToFile(path, format);
    ASSERT_TRUE(bool(streamerOr)) << llvm::toString(streamerOr.takeError());
    auto streamer = std::move(*streamerOr);

    MLIRContext context;
    Location loc = UnknownLoc::get(&context);
    constexpr unsigned numThreads = 4, perThread = 64;
    std::vector<remark::detail::Remark> remarks;
    llvm::StringSet<> remaining;
    for (unsigned i = 0; i < numThreads * perThread; ++i) {
      std::string name = "Remark" + std::to_string(i);
      remaining.insert(name);
      remarks.emplace_back(
          remark::RemarkKind::RemarkPassed, DiagnosticSeverity::Remark, loc,
          remark::RemarkOpts::name(name).category("Opt").function("kernel"));
      remarks.back().insert(std::string(128, 'x'));
    }

    std::atomic<unsigned> ready{0};
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < numThreads; ++t)
      threads.emplace_back([&, t] {
        ++ready;
        while (ready.load() != numThreads)
          std::this_thread::yield();
        for (unsigned i = 0; i < perThread; ++i)
          streamer->streamOptimizationRemark(remarks[t * perThread + i]);
      });
    for (auto &thread : threads)
      thread.join();
    streamer->finalize();
    streamer.reset();

    auto buffer = llvm::MemoryBuffer::getFile(path);
    ASSERT_TRUE(bool(buffer)) << buffer.getError().message();
    auto parser =
        llvm::remarks::createRemarkParser(format, (*buffer)->getBuffer());
    ASSERT_TRUE(bool(parser)) << llvm::toString(parser.takeError());
    for (;;) {
      auto parsed = (*parser)->next();
      if (!parsed) {
        auto error = parsed.takeError();
        ASSERT_TRUE(error.isA<llvm::remarks::EndOfFileError>())
            << llvm::toString(std::move(error));
        llvm::consumeError(std::move(error));
        break;
      }
      EXPECT_EQ((*parsed)->PassName, "Opt");
      EXPECT_EQ((*parsed)->FunctionName, "kernel");
      EXPECT_EQ(remaining.erase((*parsed)->RemarkName), 1u);
    }
    EXPECT_TRUE(remaining.empty());
  }
}
#endif
