//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

#include "llvm/Support/MD5.h"

#include <algorithm>
#include <fstream>
#include <vector>

using namespace lldb;
using namespace lldb_private;

namespace {

bool CalcMD5(const FileSpec &file_spec, uint64_t offset, uint64_t length,
             llvm::MD5::MD5Result &md5_result) {
  llvm::MD5 md5_hash;
  std::ifstream file(file_spec.GetPath(), std::ios::binary);
  if (!file.is_open())
    return false;

  if (offset > 0)
    file.seekg(offset, file.beg);

  std::vector<char> read_buf(4096);
  uint64_t total_read_bytes = 0;
  while (!file.eof()) {
    const uint64_t to_read =
        (length > 0) ? std::min(static_cast<uint64_t>(read_buf.size()),
                                length - total_read_bytes)
                     : read_buf.size();
    if (to_read == 0)
      break;

    file.read(&read_buf[0], to_read);
    const auto read_bytes = file.gcount();
    if (read_bytes == 0)
      break;

    md5_hash.update(llvm::StringRef(&read_buf[0], read_bytes));
    total_read_bytes += read_bytes;
  }

  md5_hash.final(md5_result);
  return true;
}

} // namespace

bool FileSystem::CalculateMD5(const FileSpec &file_spec, uint64_t &low,
                              uint64_t &high) {
  return CalculateMD5(file_spec, 0, 0, low, high);
}

bool FileSystem::CalculateMD5(const FileSpec &file_spec, uint64_t offset,
                              uint64_t length, uint64_t &low, uint64_t &high) {
  llvm::MD5::MD5Result md5_result;
  if (!CalcMD5(file_spec, offset, length, md5_result))
    return false;

  const auto uint64_res = reinterpret_cast<const uint64_t *>(md5_result);
  high = uint64_res[0];
  low = uint64_res[1];

  return true;
}

bool FileSystem::CalculateMD5AsString(const FileSpec &file_spec,
                                      std::string &digest_str) {
  return CalculateMD5AsString(file_spec, 0, 0, digest_str);
}

bool FileSystem::CalculateMD5AsString(const FileSpec &file_spec,
                                      uint64_t offset, uint64_t length,
                                      std::string &digest_str) {
  llvm::MD5::MD5Result md5_result;
  if (!CalcMD5(file_spec, offset, length, md5_result))
    return false;

  llvm::SmallString<32> result_str;
  llvm::MD5::stringifyResult(md5_result, result_str);
  digest_str = result_str.c_str();
  return true;
}
