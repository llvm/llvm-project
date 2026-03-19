//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_DECODERTABLEEMITTER_H
#define LLVM_UTILS_TABLEGEN_DECODERTABLEEMITTER_H

#include "DecoderTree.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {

struct DecoderTableInfo {
  bool HasCheckPredicate = false;
  bool HasSoftFail = false;
};

class DecoderTableEmitter {
  DecoderTableInfo &TableInfo;
  formatted_raw_ostream OS;

  /// The number of positions occupied by the index in the output. Used to
  /// right-align indices and left-align the text that follows them.
  unsigned IndexWidth;

  /// The current position in the output stream. After the table is emitted,
  /// this is its size.
  unsigned CurrentIndex;

  /// The index of the first byte of the table row. Used as a label in the
  /// comment following the row.
  unsigned LineStartIndex;

public:
  DecoderTableEmitter(DecoderTableInfo &TableInfo, raw_ostream &OS)
      : TableInfo(TableInfo), OS(OS) {}

  void emitTable(StringRef TableName, unsigned BitWidth,
                 const DecoderTreeNode *Root);

private:
  unsigned computeNodeSize(const DecoderTreeNode *Node) const;
  unsigned computeTableSize(const DecoderTreeNode *Root,
                            unsigned BitWidth) const;

  void emitStartLine();
  void emitOpcode(StringRef Name);
  void emitByte(uint8_t Val);
  void emitUInt8(unsigned Val);
  void emitULEB128(uint64_t Val);
  raw_ostream &emitComment(indent Indent);

  void emitCheckAnyNode(const CheckAnyNode *N, indent Indent);
  void emitCheckAllNode(const CheckAllNode *N, indent Indent);
  void emitSwitchFieldNode(const SwitchFieldNode *N, indent Indent);
  void emitCheckFieldNode(const CheckFieldNode *N, indent Indent);
  void emitCheckPredicateNode(const CheckPredicateNode *N, indent Indent);
  void emitSoftFailNode(const SoftFailNode *N, indent Indent);
  void emitDecodeNode(const DecodeNode *N, indent Indent);
  void emitNode(const DecoderTreeNode *N, indent Indent);
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_DECODERTABLEEMITTER_H
