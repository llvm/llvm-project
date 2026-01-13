//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Format.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename FormatTy>
std::string printToString(unsigned MaxN, FormatTy &&Fmt) {
  std::vector<char> Dst(MaxN + 2);
  int N = Fmt.snprint(Dst.data(), Dst.size());
  Dst.back() = 0;
  return N < 0 ? "" : Dst.data();
}

template <typename Expected, typename Arg>
constexpr bool checkDecayTypeEq(const Arg &arg) {
  return std::is_same_v<detail::decay_if_c_char_array_t<Arg>, Expected>;
}

TEST(Format, DecayIfCCharArray) {
  char Array[] = "Array";
  const char ConstArray[] = "ConstArray";
  char PtrBuf[] = "Ptr";
  char *Ptr = PtrBuf;
  const char *PtrToConst = "PtrToConst";

  EXPECT_EQ("        Literal", printToString(20, format("%15s", "Literal")));
  EXPECT_EQ("          Array", printToString(20, format("%15s", Array)));
  EXPECT_EQ("     ConstArray", printToString(20, format("%15s", ConstArray)));
  EXPECT_EQ("            Ptr", printToString(20, format("%15s", Ptr)));
  EXPECT_EQ("     PtrToConst", printToString(20, format("%15s", PtrToConst)));

  EXPECT_TRUE(checkDecayTypeEq<const char *>("Literal"));
  EXPECT_TRUE(checkDecayTypeEq<const char *>(Array));
  EXPECT_TRUE(checkDecayTypeEq<const char *>(ConstArray));
  EXPECT_TRUE(checkDecayTypeEq<char *>(Ptr));
  EXPECT_TRUE(checkDecayTypeEq<const char *>(PtrToConst));
  EXPECT_TRUE(checkDecayTypeEq<char>(PtrToConst[0]));
  EXPECT_TRUE(
      checkDecayTypeEq<const char *>(static_cast<const char *>("Literal")));

  wchar_t WCharArray[] = L"WCharArray";
  EXPECT_TRUE(checkDecayTypeEq<wchar_t[11]>(WCharArray));
  EXPECT_TRUE(checkDecayTypeEq<wchar_t>(WCharArray[0]));
}

} // namespace
