//===-- DIContext.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines DIContext, an abstract data structure that holds
// debug information data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DICONTEXT_H
#define LLVM_DEBUGINFO_DICONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class raw_ostream;

/// DILineInfo - a format-neutral container for source line information.
class DILineInfo {
  SmallString<16> FileName;
  SmallString<16> FunctionName;
  uint32_t Line;
  uint32_t Column;
public:
  DILineInfo()
    : FileName("<invalid>"), FunctionName("<invalid>"),
      Line(0), Column(0) {}
  DILineInfo(const SmallString<16> &fileName,
             const SmallString<16> &functionName,
             uint32_t line, uint32_t column)
    : FileName(fileName), FunctionName(functionName),
      Line(line), Column(column) {}

  const char *getFileName() { return FileName.c_str(); }
  const char *getFunctionName() { return FunctionName.c_str(); }
  uint32_t getLine() const { return Line; }
  uint32_t getColumn() const { return Column; }

  bool operator==(const DILineInfo &RHS) const {
    return Line == RHS.Line && Column == RHS.Column &&
           FileName.equals(RHS.FileName) &&
           FunctionName.equals(RHS.FunctionName);
  }
  bool operator!=(const DILineInfo &RHS) const {
    return !(*this == RHS);
  }
};

/// DIInliningInfo - a format-neutral container for inlined code description.
class DIInliningInfo {
  SmallVector<DILineInfo, 4> Frames;
 public:
  DIInliningInfo() {}
  DILineInfo getFrame(unsigned Index) const {
    assert(Index < Frames.size());
    return Frames[Index];
  }
  uint32_t getNumberOfFrames() const {
    return Frames.size();
  }
  void addFrame(const DILineInfo &Frame) {
    Frames.push_back(Frame);
  }
};

/// DILineInfoSpecifier - controls which fields of DILineInfo container
/// should be filled with data.
class DILineInfoSpecifier {
  const uint32_t Flags;  // Or'ed flags that set the info we want to fetch.
public:
  enum Specification {
    FileLineInfo = 1 << 0,
    AbsoluteFilePath = 1 << 1,
    FunctionName = 1 << 2
  };
  // Use file/line info by default.
  DILineInfoSpecifier(uint32_t flags = FileLineInfo) : Flags(flags) {}
  bool needs(Specification spec) const {
    return (Flags & spec) > 0;
  }
};

// In place of applying the relocations to the data we've read from disk we use
// a separate mapping table to the side and checking that at locations in the
// dwarf where we expect relocated values. This adds a bit of complexity to the
// dwarf parsing/extraction at the benefit of not allocating memory for the
// entire size of the debug info sections.
typedef DenseMap<uint64_t, std::pair<uint8_t, int64_t> > RelocAddrMap;

class DIContext {
public:
  virtual ~DIContext();

  /// getDWARFContext - get a context for binary DWARF data.
  static DIContext *getDWARFContext(bool isLittleEndian,
                                    StringRef infoSection,
                                    StringRef abbrevSection,
                                    StringRef aRangeSection = StringRef(),
                                    StringRef lineSection = StringRef(),
                                    StringRef stringSection = StringRef(),
                                    StringRef rangeSection = StringRef(),
                                    const RelocAddrMap &Map = RelocAddrMap());

  virtual void dump(raw_ostream &OS) = 0;

  virtual DILineInfo getLineInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;
  virtual DIInliningInfo getInliningInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;
};

}

#endif
