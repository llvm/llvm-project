//===----------- RPCUtilsTest.cpp - Unit tests the Orc RPC utils ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RPCChannel.h"
#include "llvm/ExecutionEngine/Orc/RPCUtils.h"
#include "gtest/gtest.h"

#include <queue>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::remote;

class QueueChannel : public RPCChannel {
public:
  QueueChannel(std::queue<char> &Queue) : Queue(Queue) {}

  std::error_code readBytes(char *Dst, unsigned Size) override {
    while (Size--) {
      *Dst++ = Queue.front();
      Queue.pop();
    }
    return std::error_code();
  }

  std::error_code appendBytes(const char *Src, unsigned Size) override {
    while (Size--)
      Queue.push(*Src++);
    return std::error_code();
  }

  std::error_code send() override { return std::error_code(); }

private:
  std::queue<char> &Queue;
};

class DummyRPC : public testing::Test,
                 public RPC<QueueChannel> {
public:
  typedef Procedure<1, bool> Proc1;
  typedef Procedure<2, int8_t,
                       uint8_t,
                       int16_t,
                       uint16_t,
                       int32_t,
                       uint32_t,
                       int64_t,
                       uint64_t,
                       bool,
                       std::string,
                       std::vector<int>> AllTheTypes;
};


TEST_F(DummyRPC, TestBasic) {
  std::queue<char> Queue;
  QueueChannel C(Queue);

  {
    // Make a call to Proc1.
    auto EC = call<Proc1>(C, true);
    EXPECT_FALSE(EC) << "Simple call over queue failed";
  }

  {
    // Expect a call to Proc1.
    auto EC = expect<Proc1>(C,
                [&](bool &B) {
                  EXPECT_EQ(B, true)
                    << "Bool serialization broken";
                  return std::error_code();
                });
    EXPECT_FALSE(EC) << "Simple expect over queue failed";
  }
}

TEST_F(DummyRPC, TestSerialization) {
  std::queue<char> Queue;
  QueueChannel C(Queue);

  {
    // Make a call to Proc1.
    std::vector<int> v({42, 7});
    auto EC = call<AllTheTypes>(C,
                                -101,
                                250,
                                -10000,
                                10000,
                                -1000000000,
                                1000000000,
                                -10000000000,
                                10000000000,
                                true,
                                "foo",
                                v);
    EXPECT_FALSE(EC) << "Big (serialization test) call over queue failed";
  }

  {
    // Expect a call to Proc1.
    auto EC = expect<AllTheTypes>(C,
                [&](int8_t &s8,
                    uint8_t &u8,
                    int16_t &s16,
                    uint16_t &u16,
                    int32_t &s32,
                    uint32_t &u32,
                    int64_t &s64,
                    uint64_t &u64,
                    bool &b,
                    std::string &s,
                    std::vector<int> &v) {

                    EXPECT_EQ(s8, -101)
                      << "int8_t serialization broken";
                    EXPECT_EQ(u8, 250)
                      << "uint8_t serialization broken";
                    EXPECT_EQ(s16, -10000)
                      << "int16_t serialization broken";
                    EXPECT_EQ(u16, 10000)
                      << "uint16_t serialization broken";
                    EXPECT_EQ(s32, -1000000000)
                      << "int32_t serialization broken";
                    EXPECT_EQ(u32, 1000000000ULL)
                      << "uint32_t serialization broken";
                    EXPECT_EQ(s64, -10000000000)
                      << "int64_t serialization broken";
                    EXPECT_EQ(u64, 10000000000ULL)
                      << "uint64_t serialization broken";
                    EXPECT_EQ(b, true)
                      << "bool serialization broken";
                    EXPECT_EQ(s, "foo")
                      << "std::string serialization broken";
                    EXPECT_EQ(v, std::vector<int>({42, 7}))
                      << "std::vector serialization broken";
                    return std::error_code();
                  });
    EXPECT_FALSE(EC) << "Big (serialization test) call over queue failed";
  }
}
