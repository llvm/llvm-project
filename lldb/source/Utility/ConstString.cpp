//===-- ConstString.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ConstString.h"

#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StringPool.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/FormatProviders.h"

#include <utility>

#include <cinttypes>
#include <cstdint>
#include <cstring>

using namespace lldb_private;

ConstString::ConstString(const char *cstr)
    : m_string(StringPool::GetGlobal().GetConstCString(cstr)) {}

ConstString::ConstString(const char *cstr, size_t cstr_len)
    : m_string(
          StringPool::GetGlobal().GetConstCStringWithLength(cstr, cstr_len)) {}

ConstString::ConstString(llvm::StringRef s)
    : m_string(StringPool::GetGlobal().Intern(s)) {}

bool ConstString::operator<(ConstString rhs) const {
  if (m_string == rhs.m_string)
    return false;

  llvm::StringRef lhs_string_ref(GetStringRef());
  llvm::StringRef rhs_string_ref(rhs.GetStringRef());

  // If both have valid C strings, then return the comparison
  if (lhs_string_ref.data() && rhs_string_ref.data())
    return lhs_string_ref < rhs_string_ref;

  // Else one of them was nullptr, so if LHS is nullptr then it is less than
  return lhs_string_ref.data() == nullptr;
}

Stream &lldb_private::operator<<(Stream &s, ConstString str) {
  const char *cstr = str.GetCString();
  if (cstr != nullptr)
    s << cstr;

  return s;
}

size_t ConstString::GetLength() const {
  return StringPool::GetConstCStringLength(m_string);
}

bool ConstString::Equals(ConstString lhs, ConstString rhs,
                         const bool case_sensitive) {
  if (lhs.m_string == rhs.m_string)
    return true;

  // Since the pointers weren't equal, and identical ConstStrings always have
  // identical pointers, the result must be false for case sensitive equality
  // test.
  if (case_sensitive)
    return false;

  // perform case insensitive equality test
  llvm::StringRef lhs_string_ref(lhs.GetStringRef());
  llvm::StringRef rhs_string_ref(rhs.GetStringRef());
  return lhs_string_ref.equals_insensitive(rhs_string_ref);
}

int ConstString::Compare(ConstString lhs, ConstString rhs,
                         const bool case_sensitive) {
  // If the iterators are the same, this is the same string
  const char *lhs_cstr = lhs.m_string;
  const char *rhs_cstr = rhs.m_string;
  if (lhs_cstr == rhs_cstr)
    return 0;
  if (lhs_cstr && rhs_cstr) {
    llvm::StringRef lhs_string_ref(lhs.GetStringRef());
    llvm::StringRef rhs_string_ref(rhs.GetStringRef());

    if (case_sensitive) {
      return lhs_string_ref.compare(rhs_string_ref);
    } else {
      return lhs_string_ref.compare_insensitive(rhs_string_ref);
    }
  }

  if (lhs_cstr)
    return +1; // LHS isn't nullptr but RHS is
  else
    return -1; // LHS is nullptr but RHS isn't
}

void ConstString::Dump(Stream *s, const char *fail_value) const {
  if (s != nullptr) {
    const char *cstr = AsCString(fail_value);
    if (cstr != nullptr)
      s->PutCString(cstr);
  }
}

void ConstString::DumpDebug(Stream *s) const {
  const char *cstr = GetCString();
  size_t cstr_len = GetLength();
  // Only print the parens if we have a non-nullptr string
  const char *parens = cstr ? "\"" : "";
  s->Printf("%*p: ConstString, string = %s%s%s, length = %" PRIu64,
            static_cast<int>(sizeof(void *) * 2),
            static_cast<const void *>(this), parens, cstr, parens,
            static_cast<uint64_t>(cstr_len));
}

void ConstString::SetCString(const char *cstr) {
  m_string = StringPool::GetGlobal().GetConstCString(cstr);
}

void ConstString::SetString(llvm::StringRef s) {
  m_string = StringPool::GetGlobal().Intern(s);
}

void ConstString::SetStringWithMangledCounterpart(llvm::StringRef demangled,
                                                  ConstString mangled) {
  m_string = StringPool::GetGlobal().GetConstCStringAndSetMangledCounterPart(
      demangled, mangled.GetStringRef());
}

bool ConstString::GetMangledCounterpart(ConstString &counterpart) const {
  counterpart.m_string =
      StringPool::GetGlobal().GetMangledCounterpart(GetStringRef());
  return (bool)counterpart;
}

void ConstString::SetCStringWithLength(const char *cstr, size_t cstr_len) {
  m_string = StringPool::GetGlobal().GetConstCStringWithLength(cstr, cstr_len);
}

void ConstString::SetTrimmedCStringWithLength(const char *cstr,
                                              size_t cstr_len) {
  m_string =
      StringPool::GetGlobal().GetConstTrimmedCStringWithLength(cstr, cstr_len);
}

ConstString::MemoryStats ConstString::GetMemoryStats() {
  return StringPool::GetGlobal().GetMemoryStats();
}

void llvm::format_provider<ConstString>::format(const ConstString &CS,
                                                llvm::raw_ostream &OS,
                                                llvm::StringRef Options) {
  format_provider<StringRef>::format(CS.GetStringRef(), OS, Options);
}
