//===- ExplainOutputStyle.h ----------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_EXPLAINOUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_EXPLAINOUTPUTSTYLE_H

#include "LinePrinter.h"
#include "OutputStyle.h"

#include <string>

namespace llvm {

namespace pdb {

class DbiStream;
class PDBFile;

class ExplainOutputStyle : public OutputStyle {

public:
  ExplainOutputStyle(PDBFile &File, uint64_t FileOffset);

  Error dump() override;

private:
  bool explainBlockStatus();

  bool isFpm1() const;
  bool isFpm2() const;

  bool isSuperBlock() const;
  bool isFpmBlock() const;
  bool isBlockMapBlock() const;
  bool isStreamDirectoryBlock() const;
  Optional<uint32_t> getBlockStreamIndex() const;

  void explainSuperBlockOffset();
  void explainFpmBlockOffset();
  void explainBlockMapOffset();
  void explainStreamDirectoryOffset();
  void explainStreamOffset(uint32_t Stream);
  void explainUnknownBlock();

  void explainDbiStream(uint32_t StreamIdx, uint32_t OffsetInStream);
  void explainPdbStream(uint32_t StreamIdx, uint32_t OffsetInStream);

  PDBFile &File;
  const uint64_t FileOffset;
  const uint64_t BlockIndex;
  const uint64_t OffsetInBlock;
  LinePrinter P;
};
} // namespace pdb
} // namespace llvm

#endif
