//===------- Offload API tests - OffloadPrint.hpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// OffloadPrint.hpp must not depend on any stream library, so it is included
// first and checked before anything else can pull a stream header in.
#include <OffloadPrint.hpp>

#if defined(LLVM_SUPPORT_RAW_OSTREAM_H)
#error "OffloadPrint.hpp must not include llvm/Support/raw_ostream.h"
#endif
#if defined(_GLIBCXX_OSTREAM) || defined(_GLIBCXX_IOSFWD) ||                   \
    defined(_LIBCPP_OSTREAM) || defined(_LIBCPP_IOSFWD)
#error "OffloadPrint.hpp must not include <ostream> or <iosfwd>"
#endif

#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <charconv>
#include <cstdint>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>

// A user-defined sink with free inserts that return void. Not detected
// structurally; opted in below.
struct UserSink {
  std::string Str;
};
void operator<<(UserSink &S, const char *V) { S.Str += V; }
void operator<<(UserSink &S, char V) { S.Str += V; }
void operator<<(UserSink &S, const void *V) { S.Str += V ? "<ptr>" : "<null>"; }
void operator<<(UserSink &S, bool V) { S.Str += V ? "1" : "0"; }
void operator<<(UserSink &S, unsigned int V) { S.Str += std::to_string(V); }
void operator<<(UserSink &S, unsigned long V) { S.Str += std::to_string(V); }
void operator<<(UserSink &S, unsigned long long V) {
  S.Str += std::to_string(V);
}

// Querying is_stream<UserSink> here would instantiate it and make the
// specialization below ill-formed, so check the structural traits directly.
static_assert(!ol_print::detail::is_std_ostream_like<UserSink>::value);
static_assert(!ol_print::detail::is_raw_ostream_like<UserSink>::value);
template <> struct ol_print::is_stream<UserSink> : std::true_type {};

// Detected.
static_assert(ol_print::is_stream_v<std::ostream>);
static_assert(ol_print::is_stream_v<std::ostringstream>);
static_assert(ol_print::is_stream_v<llvm::raw_ostream>);
static_assert(ol_print::is_stream_v<llvm::raw_string_ostream>);
// Opted in.
static_assert(ol_print::is_stream_v<UserSink>);
// Rejected: not streams, or forwarding wrappers around a stream.
static_assert(!ol_print::is_stream_v<int>);
static_assert(!ol_print::is_stream_v<std::string>);
static_assert(!ol_print::is_stream_v<testing::Message>);
static_assert(!ol_print::is_stream_v<llvm::WithColor>);

template <typename T> std::string printStd(const T &V) {
  std::ostringstream OS;
  OS << V;
  return OS.str();
}

template <typename T> std::string printLLVM(const T &V) {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << V;
  return Str;
}

// Expects a pointer. Its format is up to the stream (MSVC's std::ostream prints
// 0000000000001000 where raw_ostream prints 0x1000), so the whole alphanumeric
// word is consumed and must be hex, optionally 0x-prefixed, equal to Value.
struct hex_string_t {
  const void *Value;
};

// Matches a printed string piece by piece from the front, adding a test
// failure at the first mismatch. Anything left unmatched at the end fails too.
class OutputMatcher {
public:
  explicit OutputMatcher(std::string_view Str) : Str(Str), Rest(Str) {}
  ~OutputMatcher() {
    if (!Failed && !Rest.empty())
      fail("unexpected trailing \"" + std::string(Rest) + "\"");
  }

  OutputMatcher &expectNext(std::string_view Expected) {
    if (Failed)
      return *this;
    if (Rest.substr(0, Expected.size()) != Expected)
      fail("expected \"" + std::string(Expected) + "\"");
    else
      Rest.remove_prefix(Expected.size());
    return *this;
  }

  OutputMatcher &expectNext(hex_string_t Expected) {
    if (Failed)
      return *this;
    size_t Len = 0;
    while (Len < Rest.size() &&
           std::isalnum(static_cast<unsigned char>(Rest[Len])))
      ++Len;
    std::string_view Word = Rest.substr(0, Len);
    std::string_view Digits = Word;
    if (Digits.substr(0, 2) == "0x" || Digits.substr(0, 2) == "0X")
      Digits.remove_prefix(2);
    uintptr_t Value;
    auto [End, Err] = std::from_chars(Digits.data(),
                                      Digits.data() + Digits.size(), Value, 16);
    uintptr_t ExpectedValue = reinterpret_cast<uintptr_t>(Expected.Value);
    if (Err != std::errc() || End != Digits.data() + Digits.size() ||
        Value != ExpectedValue)
      fail("expected hex for " + std::to_string(ExpectedValue) + ", got \"" +
           std::string(Word) + "\"");
    else
      Rest.remove_prefix(Len);
    return *this;
  }

private:
  void fail(const std::string &What) {
    Failed = true;
    ADD_FAILURE() << What << " at offset " << Str.size() - Rest.size()
                  << " of \"" << Str << "\"";
  }

  std::string_view Str;
  std::string_view Rest;
  bool Failed = false;
};

TEST(OffloadPrintTest, OutputMatcherHex) {
  static const hex_string_t P{reinterpret_cast<const void *>(0xab)};
  OutputMatcher("0xab").expectNext(P);
  OutputMatcher("00000000000000AB").expectNext(P);
  EXPECT_NONFATAL_FAILURE(OutputMatcher("0xac").expectNext(P), "got \"0xac\"");
  EXPECT_NONFATAL_FAILURE(OutputMatcher("0xabg").expectNext(P),
                          "got \"0xabg\"");
  EXPECT_NONFATAL_FAILURE(OutputMatcher("0x").expectNext(P), "got \"0x\"");
  EXPECT_NONFATAL_FAILURE(OutputMatcher("0xab)").expectNext(P),
                          "unexpected trailing \")\"");
}

TEST(OffloadPrintTest, Enum) {
  EXPECT_EQ(printStd(OL_ERRC_INVALID_DEVICE), "OL_ERRC_INVALID_DEVICE");
  EXPECT_EQ(printLLVM(OL_ERRC_INVALID_DEVICE), "OL_ERRC_INVALID_DEVICE");
  EXPECT_EQ(printStd(static_cast<ol_errc_t>(-1)), "unknown enumerator");
}

TEST(OffloadPrintTest, Struct) {
  ol_dimensions_t Dims{1, 2, 3};
  const char *Expected = "(struct ol_dimensions_t){.x = 1, .y = 2, .z = 3}";
  EXPECT_EQ(printStd(Dims), Expected);
  EXPECT_EQ(printLLVM(Dims), Expected);
}

TEST(OffloadPrintTest, ParamStructWithTaggedValue) {
  ol_device_handle_t Device = reinterpret_cast<ol_device_handle_t>(0x1000);
  ol_device_info_t PropName = OL_DEVICE_INFO_NAME;
  size_t PropSize = 6;
  char Name[] = "Hello";
  void *PropValue = Name;
  ol_get_device_info_params_t Params{&Device, &PropName, &PropSize, &PropValue};

  for (const std::string &Out : {printStd(&Params), printLLVM(&Params)})
    OutputMatcher(Out)
        .expectNext(".Device = ")
        .expectNext(hex_string_t{Device})
        .expectNext(", .PropName = OL_DEVICE_INFO_NAME, .PropSize = 6, "
                    ".PropValue = ")
        .expectNext(hex_string_t{Name})
        .expectNext(" (Hello)");
}

TEST(OffloadPrintTest, GTestMessage) {
  EXPECT_EQ((testing::Message() << OL_ERRC_INVALID_DEVICE).GetString(),
            "OL_ERRC_INVALID_DEVICE");
}

TEST(OffloadPrintTest, UserSink) {
  ol_platform_info_t PropName = OL_PLATFORM_INFO_NAME;
  size_t PropSize = 4;
  char Name[] = "abc";
  void *PropValue = Name;
  ol_platform_handle_t Platform = nullptr;
  ol_get_platform_info_params_t Params{&Platform, &PropName, &PropSize,
                                       &PropValue};

  UserSink S;
  S << &Params;
  EXPECT_EQ(S.Str, ".Platform = nullptr, .PropName = OL_PLATFORM_INFO_NAME, "
                   ".PropSize = 4, .PropValue = <ptr> (abc)");
}
