//===-- Checksum.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Checksum.h"
#include "llvm/ADT/SmallString.h"

using namespace lldb_private;

Checksum::Checksum(llvm::MD5::MD5Result md5) { SetMD5(md5); }

Checksum::Checksum(const Checksum &checksum) { SetMD5(checksum.m_checksum); }

Checksum &Checksum::operator=(const Checksum &checksum) {
  SetMD5(checksum.m_checksum);
  return *this;
}

void Checksum::SetMD5(llvm::MD5::MD5Result md5) {
  std::uninitialized_copy_n(md5.begin(), 16, m_checksum.begin());
}

Checksum::operator bool() const {
  return !std::equal(m_checksum.begin(), m_checksum.end(), sentinel.begin());
}

bool Checksum::operator==(const Checksum &checksum) const {
  return std::equal(m_checksum.begin(), m_checksum.end(),
                    checksum.m_checksum.begin());
}

bool Checksum::operator!=(const Checksum &checksum) const {
  return !std::equal(m_checksum.begin(), m_checksum.end(),
                     checksum.m_checksum.begin());
}

std::string Checksum::digest() const {
  return std::string(m_checksum.digest().str());
}

llvm::MD5::MD5Result Checksum::sentinel = {0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0};
