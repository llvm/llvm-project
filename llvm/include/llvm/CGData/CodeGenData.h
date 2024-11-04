//===- CodeGenData.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for codegen data that has stable summary which
// can be used to optimize the code in the subsequent codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CGDATA_CODEGENDATA_H
#define LLVM_CGDATA_CODEGENDATA_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CGData/OutlinedHashTree.h"
#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <mutex>

namespace llvm {

enum CGDataSectKind {
#define CG_DATA_SECT_ENTRY(Kind, SectNameCommon, SectNameCoff, Prefix) Kind,
#include "llvm/CGData/CodeGenData.inc"
};

std::string getCodeGenDataSectionName(CGDataSectKind CGSK,
                                      Triple::ObjectFormatType OF,
                                      bool AddSegmentInfo = true);

enum class CGDataKind {
  Unknown = 0x0,
  // A function outlining info.
  FunctionOutlinedHashTree = 0x1,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/FunctionOutlinedHashTree)
};

const std::error_category &cgdata_category();

enum class cgdata_error {
  success = 0,
  eof,
  bad_magic,
  bad_header,
  empty_cgdata,
  malformed,
  unsupported_version,
};

inline std::error_code make_error_code(cgdata_error E) {
  return std::error_code(static_cast<int>(E), cgdata_category());
}

class CGDataError : public ErrorInfo<CGDataError> {
public:
  CGDataError(cgdata_error Err, const Twine &ErrStr = Twine())
      : Err(Err), Msg(ErrStr.str()) {
    assert(Err != cgdata_error::success && "Not an error");
  }

  std::string message() const override;

  void log(raw_ostream &OS) const override { OS << message(); }

  std::error_code convertToErrorCode() const override {
    return make_error_code(Err);
  }

  cgdata_error get() const { return Err; }
  const std::string &getMessage() const { return Msg; }

  /// Consume an Error and return the raw enum value contained within it, and
  /// the optional error message. The Error must either be a success value, or
  /// contain a single CGDataError.
  static std::pair<cgdata_error, std::string> take(Error E) {
    auto Err = cgdata_error::success;
    std::string Msg;
    handleAllErrors(std::move(E), [&Err, &Msg](const CGDataError &IPE) {
      assert(Err == cgdata_error::success && "Multiple errors encountered");
      Err = IPE.get();
      Msg = IPE.getMessage();
    });
    return {Err, Msg};
  }

  static char ID;

private:
  cgdata_error Err;
  std::string Msg;
};

enum CGDataMode {
  None,
  Read,
  Write,
};

class CodeGenData {
  /// Global outlined hash tree that has oulined hash sequences across modules.
  std::unique_ptr<OutlinedHashTree> PublishedHashTree;

  /// This flag is set when -fcodegen-data-generate is passed.
  /// Or, it can be mutated with -fcodegen-data-thinlto-two-rounds.
  bool EmitCGData;

  /// This is a singleton instance which is thread-safe. Unlike profile data
  /// which is largely function-based, codegen data describes the whole module.
  /// Therefore, this can be initialized once, and can be used across modules
  /// instead of constructing the same one for each codegen backend.
  static std::unique_ptr<CodeGenData> Instance;
  static std::once_flag OnceFlag;

  CodeGenData() = default;

public:
  ~CodeGenData() = default;

  static CodeGenData &getInstance();

  /// Returns true if we have a valid outlined hash tree.
  bool hasOutlinedHashTree() {
    return PublishedHashTree && !PublishedHashTree->empty();
  }

  /// Returns the outlined hash tree. This can be globally used in a read-only
  /// manner.
  const OutlinedHashTree *getOutlinedHashTree() {
    return PublishedHashTree.get();
  }

  /// Returns true if we should write codegen data.
  bool emitCGData() { return EmitCGData; }

  /// Publish the (globally) merged or read outlined hash tree.
  void publishOutlinedHashTree(std::unique_ptr<OutlinedHashTree> HashTree) {
    PublishedHashTree = std::move(HashTree);
    // Ensure we disable emitCGData as we do not want to read and write both.
    EmitCGData = false;
  }
};

namespace cgdata {

inline bool hasOutlinedHashTree() {
  return CodeGenData::getInstance().hasOutlinedHashTree();
}

inline const OutlinedHashTree *getOutlinedHashTree() {
  return CodeGenData::getInstance().getOutlinedHashTree();
}

inline bool emitCGData() { return CodeGenData::getInstance().emitCGData(); }

inline void
publishOutlinedHashTree(std::unique_ptr<OutlinedHashTree> HashTree) {
  CodeGenData::getInstance().publishOutlinedHashTree(std::move(HashTree));
}

void warn(Error E, StringRef Whence = "");
void warn(Twine Message, std::string Whence = "", std::string Hint = "");

} // end namespace cgdata

namespace IndexedCGData {

// A signature for data validation, representing "\xffcgdata\x81" in
// little-endian order
const uint64_t Magic = 0x81617461646763ff;

enum CGDataVersion {
  // Version 1 is the first version. This version supports the outlined
  // hash tree.
  Version1 = 1,
  CurrentVersion = CG_DATA_INDEX_VERSION
};
const uint64_t Version = CGDataVersion::CurrentVersion;

struct Header {
  uint64_t Magic;
  uint32_t Version;
  uint32_t DataKind;
  uint64_t OutlinedHashTreeOffset;

  // New fields should only be added at the end to ensure that the size
  // computation is correct. The methods below need to be updated to ensure that
  // the new field is read correctly.

  // Reads a header struct from the buffer.
  static Expected<Header> readFromBuffer(const unsigned char *Curr);
};

} // end namespace IndexedCGData

} // end namespace llvm

#endif // LLVM_CODEGEN_PREPARE_H
