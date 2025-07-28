//===- llvm/Support/DebugLog.h - Logging like debug output ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains macros for logging like debug output. It builds upon the
// support in Debug.h but provides a utility function for common debug output
// style.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DEBUGLOG_H
#define LLVM_SUPPORT_DEBUGLOG_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
#ifndef NDEBUG

// Output with given inputs and trailing newline. E.g.,
//   LDBG() << "Bitset contains: " << Bitset;
// is equivalent to
//   LLVM_DEBUG(dbgs() << DEBUG_TYPE << " [" << __FILE__ << ":" << __LINE__
//              << "] " << "Bitset contains: " << Bitset << "\n");
#define LDBG() DEBUGLOG_WITH_STREAM_AND_TYPE(llvm::dbgs(), DEBUG_TYPE)

#define DEBUGLOG_WITH_STREAM_TYPE_AND_FILE(STREAM, TYPE, FILE)                 \
  for (bool _c = (::llvm::DebugFlag && ::llvm::isCurrentDebugType(TYPE)); _c;  \
       _c = false)                                                             \
  ::llvm::impl::raw_ldbg_ostream{                                              \
      ::llvm::impl::computePrefix(TYPE, FILE, __LINE__), (STREAM)}             \
      .asLvalue()
// When __SHORT_FILE__ is not defined, the File is the full path,
// otherwise __SHORT_FILE__ is defined in CMake to provide the file name
// without the path prefix.
#if defined(__SHORT_FILE__)
#define DEBUGLOG_WITH_STREAM_AND_TYPE(STREAM, TYPE)                            \
  DEBUGLOG_WITH_STREAM_TYPE_AND_FILE(STREAM, TYPE, __SHORT_FILE__)
#else
#define DEBUGLOG_WITH_STREAM_AND_TYPE(STREAM, TYPE)                            \
  DEBUGLOG_WITH_STREAM_TYPE_AND_FILE(STREAM, TYPE,                             \
                                     ::llvm::impl::getShortFileName(__FILE__))
#endif

namespace impl {

/// A raw_ostream that tracks `\n` and print the prefix.
class LLVM_ABI raw_ldbg_ostream final : public raw_ostream {
  std::string Prefix;
  raw_ostream &Os;
  bool HasPendingNewline = true;

  /// Split the line on newlines and insert the prefix before each newline.
  /// Forward everything to the underlying stream.
  void write_impl(const char *Ptr, size_t Size) final {
    auto Str = StringRef(Ptr, Size);
    // Handle the initial prefix.
    if (!Str.empty())
      writeWithPrefix(StringRef());

    auto Eol = Str.find('\n');
    while (Eol != StringRef::npos) {
      StringRef Line = Str.take_front(Eol + 1);
      if (!Line.empty())
        writeWithPrefix(Line);
      HasPendingNewline = true;
      Str = Str.drop_front(Eol + 1);
      Eol = Str.find('\n');
    }
    if (!Str.empty())
      writeWithPrefix(Str);
  }
  void emitPrefix() { Os.write(Prefix.c_str(), Prefix.size()); }
  void writeWithPrefix(StringRef Str) {
    if (HasPendingNewline) {
      emitPrefix();
      HasPendingNewline = false;
    }
    Os.write(Str.data(), Str.size());
  }

public:
  explicit raw_ldbg_ostream(std::string Prefix, raw_ostream &Os)
      : Prefix(std::move(Prefix)), Os(Os) {
    SetUnbuffered();
  }
  ~raw_ldbg_ostream() final {
    flushEol();
    Os << '\n';
  }
  void flushEol() {
    if (HasPendingNewline) {
      emitPrefix();
      HasPendingNewline = false;
    }
  }

  /// Forward the current_pos method to the underlying stream.
  uint64_t current_pos() const final { return Os.tell(); }

  /// Some of the `<<` operators expect an lvalue, so we trick the type system.
  raw_ldbg_ostream &asLvalue() { return *this; }
};

/// Remove the path prefix from the file name.
static LLVM_ATTRIBUTE_UNUSED constexpr const char *
getShortFileName(const char *path) {
  const char *filename = path;
  for (const char *p = path; *p != '\0'; ++p) {
    if (*p == '/' || *p == '\\')
      filename = p + 1;
  }
  return filename;
}

/// Compute the prefix for the debug log in the form of:
/// "[DebugType] File:Line "
/// Where the File is the file name without the path prefix.
static LLVM_ATTRIBUTE_UNUSED std::string
computePrefix(const char *DebugType, const char *File, int Line) {
  std::string Prefix;
  raw_string_ostream OsPrefix(Prefix);
  if (DebugType)
    OsPrefix << "[" << DebugType << "] ";
  OsPrefix << File << ":" << Line << " ";
  return OsPrefix.str();
}
} // end namespace impl
#else
// As others in Debug, When compiling without assertions, the -debug-* options
// and all inputs too LDBG() are ignored.
#define LDBG()                                                                 \
  for (bool _c = false; _c; _c = false)                                        \
  ::llvm::nulls()
#endif
} // end namespace llvm

#endif // LLVM_SUPPORT_DEBUGLOG_H
