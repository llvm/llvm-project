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
#include <gtest/gtest.h>
#include <sstream>
#include <string>

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

  std::string Std = printStd(&Params);
  std::string LLVM = printLLVM(&Params);
  EXPECT_EQ(Std, LLVM);
  EXPECT_NE(Std.find(".PropName = OL_DEVICE_INFO_NAME"), std::string::npos)
      << Std;
  EXPECT_NE(Std.find(".PropSize = 6"), std::string::npos) << Std;
  EXPECT_NE(Std.find("(Hello)"), std::string::npos) << Std;
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
