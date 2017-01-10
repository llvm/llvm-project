//===- xray-log-reader.h - XRay Log Reader Interface ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Define the interface for an XRay log reader. Currently we only support one
// version of the log (naive log) with fixed-sized records.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_XRAY_XRAY_LOG_READER_H
#define LLVM_TOOLS_LLVM_XRAY_XRAY_LOG_READER_H

#include <cstdint>
#include <deque>
#include <vector>

#include "xray-record-yaml.h"
#include "xray-record.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

namespace llvm {
namespace xray {

class LogReader {
  XRayFileHeader FileHeader;
  std::vector<XRayRecord> Records;

  typedef std::vector<XRayRecord>::const_iterator citerator;

public:
  typedef std::function<Error(StringRef, XRayFileHeader &,
                              std::vector<XRayRecord> &)>
      LoaderFunction;

  LogReader(StringRef Filename, Error &Err, bool Sort, LoaderFunction Loader);

  const XRayFileHeader &getFileHeader() const { return FileHeader; }

  citerator begin() const { return Records.begin(); }
  citerator end() const { return Records.end(); }
  size_t size() const { return Records.size(); }
};

Error NaiveLogLoader(StringRef Data, XRayFileHeader &FileHeader,
                     std::vector<XRayRecord> &Records);
Error YAMLLogLoader(StringRef Data, XRayFileHeader &FileHeader,
                    std::vector<XRayRecord> &Records);

} // namespace xray
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_XRAY_XRAY_LOG_READER_H
