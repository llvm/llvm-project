#include "DataFormatters/FormatterBytecode.h"
#include "lldb/Utility/StreamString.h"

#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using namespace FormatterBytecode;
using llvm::StringRef;

namespace {
class FormatterBytecodeTest : public ::testing::Test {};

bool Interpret(std::vector<uint8_t> code, DataStack &data) {
  auto buf =
      StringRef(reinterpret_cast<const char *>(code.data()), code.size());
  std::vector<ControlStackElement> control({buf});
  if (auto error = Interpret(control, data, sel_summary)) {
#ifndef NDEBUG
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
#else
    llvm::consumeError(std::move(error));
#endif
    return false;
  }
  return true;
}

} // namespace

TEST_F(FormatterBytecodeTest, StackOps) {
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 23, op_dup, op_plus}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 46u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_drop}, data));
    ASSERT_EQ(data.size(), 0u);
  }
  {
    for (unsigned char i = 0; i < 3; ++i) {
      DataStack data;

      ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_lit_uint, 2,
                             op_lit_uint, i, op_pick},
                            data));
      ASSERT_EQ(data.Pop<uint64_t>(), i);
    }
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_over}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_swap}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret(
        {op_lit_uint, 0, op_lit_uint, 1, op_lit_uint, 2, op_rot}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_EQ(data.Pop<uint64_t>(), 2u);
  }
}

TEST_F(FormatterBytecodeTest, ControlOps) {
  {
    DataStack data;
    ASSERT_TRUE(
        Interpret({op_lit_uint, 0, op_begin, 2, op_lit_uint, 42, op_if}, data));
    ASSERT_EQ(data.size(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(
        Interpret({op_lit_uint, 1, op_begin, 2, op_lit_uint, 42, op_if}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 42u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_begin, 2, op_lit_uint, 42,
                           op_begin, 2, op_lit_uint, 23, op_ifelse},
                          data));
    ASSERT_EQ(data.Pop<uint64_t>(), 23u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_begin, 2, op_lit_uint, 42,
                           op_begin, 2, op_lit_uint, 23, op_ifelse},
                          data));
    ASSERT_EQ(data.Pop<uint64_t>(), 42u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_begin, 3, op_lit_uint, 42,
                           op_return, op_if, op_lit_uint, 23},
                          data));
    ASSERT_EQ(data.Pop<uint64_t>(), 42u);
  }
}

TEST_F(FormatterBytecodeTest, ConversionOps) {
  {
    DataStack data(lldb::ValueObjectSP{});
    ASSERT_TRUE(Interpret({op_is_null}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 1u, op_as_int}, data));
    ASSERT_EQ(data.Pop<int64_t>(), 1);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_int, 126, op_as_uint}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), ~1ULL);
  }
}

TEST_F(FormatterBytecodeTest, ArithOps) {
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 2, op_lit_uint, 3, op_plus}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 5u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 3, op_lit_uint, 2, op_minus}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 3, op_lit_uint, 2, op_mul}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 6u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 6, op_lit_uint, 2, op_div}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 3u);
  }
  {
    DataStack data;
    ASSERT_FALSE(Interpret({op_lit_uint, 23, op_lit_uint, 0, op_div}, data));
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 2, op_shl}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 4u);
  }
  {
    DataStack data;
    unsigned char minus_one = 127;
    ASSERT_FALSE(
        Interpret({op_lit_int, minus_one, op_lit_uint, 2, op_shl}, data));
    unsigned char minus_two = 126;
    ASSERT_TRUE(
        Interpret({op_lit_int, minus_two, op_lit_uint, 1, op_shr}, data));
    ASSERT_EQ(data.Pop<int64_t>(), -1);
  }
  {
    DataStack data;
    ASSERT_FALSE(Interpret({op_lit_uint, 1, op_lit_uint, 65, op_shl}, data));
    ASSERT_FALSE(Interpret({op_lit_uint, 1, op_lit_uint, 65, op_shr}, data));
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 1, op_and}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_and}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 1, op_or}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_or}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 0, op_or}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 1, op_xor}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_xor}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 0, op_xor}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_not}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0xffffffffffffffff);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_eq}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 0, op_eq}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_neq}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 0, op_neq}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_lt}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 0, op_lt}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 1, op_lt}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_gt}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 0, op_gt}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 1, op_gt}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_le}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 0, op_le}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 1, op_le}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
  }
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 0, op_lit_uint, 1, op_ge}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 0u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 0, op_ge}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
    ASSERT_TRUE(Interpret({op_lit_uint, 1, op_lit_uint, 1, op_ge}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 1u);
  }
}

TEST_F(FormatterBytecodeTest, CallOps) {
  {
    DataStack data;
    data.Push(std::string{"hello"});
    ASSERT_TRUE(Interpret({op_lit_selector, sel_strlen, op_call}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), 5u);
  }
  {
    DataStack data;
    data.Push(std::string{"A"});
    data.Push(std::string{"B"});
    data.Push(std::string{"{1}{0}"});
    ASSERT_TRUE(Interpret({op_lit_selector, sel_fmt, op_call}, data));
    ASSERT_EQ(data.Pop<std::string>(), "BA");
  }
  {
    DataStack data;
    data.Push(std::string{"{0}"});
    ASSERT_FALSE(Interpret({op_lit_selector, sel_fmt, op_call}, data));
  }
}
