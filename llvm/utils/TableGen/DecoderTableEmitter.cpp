//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DecoderTableEmitter.h"
#include "Common/CodeGenInstruction.h"
#include "Common/InstructionEncoding.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

void DecoderTableEmitter::analyzeNode(const DecoderTreeNode *Node) const {
  switch (Node->getKind()) {
  case DecoderTreeNode::CheckAny: {
    const auto *N = static_cast<const CheckAnyNode *>(Node);
    for (const DecoderTreeNode *Child : N->children())
      analyzeNode(Child);
    break;
  }
  case DecoderTreeNode::CheckAll: {
    const auto *N = static_cast<const CheckAllNode *>(Node);
    for (const DecoderTreeNode *Child : N->children())
      analyzeNode(Child);
    break;
  }
  case DecoderTreeNode::CheckField:
    break;
  case DecoderTreeNode::SwitchField: {
    const auto *N = static_cast<const SwitchFieldNode *>(Node);
    for (const DecoderTreeNode *Child : make_second_range(N->cases()))
      analyzeNode(Child);
    break;
  }
  case DecoderTreeNode::CheckPredicate: {
    const auto *N = static_cast<const CheckPredicateNode *>(Node);
    TableInfo.insertPredicate(N->getPredicateString());
    break;
  }
  case DecoderTreeNode::SoftFail:
    break;
  case DecoderTreeNode::Decode: {
    const auto *N = static_cast<const DecodeNode *>(Node);
    TableInfo.insertDecoder(N->getDecoderString());
    break;
  }
  }
}

unsigned
DecoderTableEmitter::computeNodeSize(const DecoderTreeNode *Node) const {
  // To make the arithmetic below clearer.
  static constexpr unsigned OpcodeSize = 1;
  static constexpr unsigned FieldWidthSize = 1;

  switch (Node->getKind()) {
  case DecoderTreeNode::CheckAny: {
    const auto *N = static_cast<const CheckAnyNode *>(Node);
    // Pretend the node was optimized. See the comment in emitCheckAnyNode.
    if (range_size(N->children()) == 1)
      return computeNodeSize(*N->child_begin());
    unsigned Size = 0;
    // All children except the last one are preceded by OPC_Scope opcode and
    // the size of the child.
    for (const DecoderTreeNode *Child : drop_end(N->children())) {
      unsigned ChildSize = computeNodeSize(Child);
      Size += OpcodeSize + getULEB128Size(ChildSize) + ChildSize;
    }
    const DecoderTreeNode *Child = *std::prev(N->child_end());
    return Size + computeNodeSize(Child);
  }
  case DecoderTreeNode::CheckAll: {
    const auto *N = static_cast<const CheckAllNode *>(Node);
    unsigned Size = 0;
    for (const DecoderTreeNode *Child : N->children())
      Size += computeNodeSize(Child);
    return Size;
  }
  case DecoderTreeNode::CheckField: {
    const auto *N = static_cast<const CheckFieldNode *>(Node);
    return OpcodeSize + getULEB128Size(N->getStartBit()) + FieldWidthSize +
           getULEB128Size(N->getValue());
  }
  case DecoderTreeNode::SwitchField: {
    const auto *N = static_cast<const SwitchFieldNode *>(Node);
    unsigned Size =
        OpcodeSize + getULEB128Size(N->getStartBit()) + FieldWidthSize;

    for (auto [Val, Child] : drop_end(N->cases())) {
      unsigned ChildSize = computeNodeSize(Child);
      Size += getULEB128Size(Val) + getULEB128Size(ChildSize) + ChildSize;
    }

    auto [Val, Child] = *std::prev(N->case_end());
    unsigned ChildSize = computeNodeSize(Child);
    Size += getULEB128Size(Val) + getULEB128Size(0) + ChildSize;
    return Size;
  }
  case DecoderTreeNode::CheckPredicate: {
    const auto *N = static_cast<const CheckPredicateNode *>(Node);
    unsigned PredicateIndex =
        TableInfo.getPredicateIndex(N->getPredicateString());
    return OpcodeSize + getULEB128Size(PredicateIndex);
  }
  case DecoderTreeNode::SoftFail: {
    const auto *N = static_cast<const SoftFailNode *>(Node);
    return OpcodeSize + getULEB128Size(N->getPositiveMask()) +
           getULEB128Size(N->getNegativeMask());
  }
  case DecoderTreeNode::Decode: {
    const auto *N = static_cast<const DecodeNode *>(Node);
    unsigned InstOpcode = N->getEncoding().getInstruction()->EnumVal;
    unsigned DecoderIndex = TableInfo.getDecoderIndex(N->getDecoderString());
    return OpcodeSize + getULEB128Size(InstOpcode) +
           getULEB128Size(DecoderIndex);
  }
  }
  llvm_unreachable("Unknown node kind");
}

unsigned DecoderTableEmitter::computeTableSize(const DecoderTreeNode *Root,
                                               unsigned BitWidth) const {
  unsigned Size = 0;
  if (BitWidth)
    Size += getULEB128Size(BitWidth);
  Size += computeNodeSize(Root);
  return Size;
}

void DecoderTableEmitter::emitStartLine() {
  CommentIndex = CurrentIndex;
  OS.indent(2);
}

void DecoderTableEmitter::emitOpcode(StringRef Name) {
  emitStartLine();
  OS << Name << ", ";
  ++CurrentIndex;
}

void DecoderTableEmitter::emitByte(uint8_t Val) {
  OS << static_cast<unsigned>(Val) << ", ";
  ++CurrentIndex;
}

void DecoderTableEmitter::emitUInt8(unsigned Val) {
  assert(isUInt<8>(Val));
  emitByte(Val);
}

void DecoderTableEmitter::emitULEB128(uint64_t Val) {
  while (Val >= 0x80) {
    emitByte((Val & 0x7F) | 0x80);
    Val >>= 7;
  }
  emitByte(Val);
}

formatted_raw_ostream &DecoderTableEmitter::emitComment(indent Indent) {
  constexpr unsigned CommentColumn = 45;
  if (OS.getColumn() > CommentColumn)
    OS << '\n';
  OS.PadToColumn(CommentColumn);
  OS << "// " << format_decimal(CommentIndex, IndexWidth) << ": " << Indent;
  return OS;
}

void DecoderTableEmitter::emitCheckAnyNode(const CheckAnyNode *N,
                                           indent Indent) {
  // TODO: Single-child CheckAny node should be optimized out. For now,
  //   pretend this is the case and print the single child unindented.
  if (range_size(N->children()) == 1) {
    emitNode(*N->child_begin(), Indent);
    return;
  }

  for (const DecoderTreeNode *Child : drop_end(N->children())) {
    emitOpcode("OPC_Scope");
    emitULEB128(computeNodeSize(Child));

    emitComment(Indent) << "{\n";
    emitNode(Child, Indent + 1);
    emitComment(Indent) << "}\n";
  }

  const DecoderTreeNode *Child = *std::prev(N->child_end());
  emitComment(Indent) << "{\n";
  emitNode(Child, Indent + 1);
  emitComment(Indent) << "}\n";
}

void DecoderTableEmitter::emitCheckAllNode(const CheckAllNode *N,
                                           indent Indent) {
  for (const DecoderTreeNode *Child : N->children())
    emitNode(Child, Indent);
}

void DecoderTableEmitter::emitSwitchFieldNode(const SwitchFieldNode *N,
                                              indent Indent) {
  unsigned LSB = N->getStartBit();
  unsigned Width = N->getNumBits();
  unsigned MSB = LSB + Width - 1;

  emitOpcode("OPC_SwitchField");
  emitULEB128(LSB);
  emitUInt8(Width);

  emitComment(Indent) << "switch Inst[" << MSB << ':' << LSB << "] {\n";

  for (auto [Val, Child] : drop_end(N->cases())) {
    emitStartLine();
    emitULEB128(Val);
    emitULEB128(computeNodeSize(Child));

    emitComment(Indent) << "case " << format_hex(Val, 0) << ": {\n";
    emitNode(Child, Indent + 1);
    emitComment(Indent) << "}\n";
  }

  auto [Val, Child] = *std::prev(N->case_end());
  emitStartLine();
  emitULEB128(Val);
  emitULEB128(0);

  emitComment(Indent) << "case " << format_hex(Val, 0) << ": {\n";
  emitNode(Child, Indent + 1);
  emitComment(Indent) << "}\n";

  emitComment(Indent) << "} // switch Inst[" << MSB << ':' << LSB << "]\n";
}

void DecoderTableEmitter::emitCheckFieldNode(const CheckFieldNode *N,
                                             indent Indent) {
  unsigned LSB = N->getStartBit();
  unsigned Width = N->getNumBits();
  unsigned MSB = LSB + Width - 1;
  uint64_t Val = N->getValue();

  emitOpcode("OPC_CheckField");
  emitULEB128(LSB);
  emitUInt8(Width);
  emitULEB128(Val);

  emitComment(Indent);
  OS << "check Inst[" << MSB << ':' << LSB << "] == " << format_hex(Val, 0)
     << '\n';
}

void DecoderTableEmitter::emitCheckPredicateNode(const CheckPredicateNode *N,
                                                 indent Indent) {
  unsigned PredicateIndex =
      TableInfo.getPredicateIndex(N->getPredicateString());

  emitOpcode("OPC_CheckPredicate");
  emitULEB128(PredicateIndex);
  TableInfo.HasCheckPredicate = true;

  emitComment(Indent) << "check predicate " << PredicateIndex << "\n";
}

void DecoderTableEmitter::emitSoftFailNode(const SoftFailNode *N,
                                           indent Indent) {
  uint64_t PositiveMask = N->getPositiveMask();
  uint64_t NegativeMask = N->getNegativeMask();

  emitOpcode("OPC_SoftFail");
  emitULEB128(PositiveMask);
  emitULEB128(NegativeMask);
  TableInfo.HasSoftFail = true;

  emitComment(Indent) << "check softfail";
  OS << " pos=" << format_hex(PositiveMask, 10);
  OS << " neg=" << format_hex(NegativeMask, 10) << '\n';
}

void DecoderTableEmitter::emitDecodeNode(const DecodeNode *N, indent Indent) {
  const InstructionEncoding &Encoding = N->getEncoding();
  unsigned InstOpcode = Encoding.getInstruction()->EnumVal;
  unsigned DecoderIndex = TableInfo.getDecoderIndex(N->getDecoderString());

  emitOpcode("OPC_Decode");
  emitULEB128(InstOpcode);
  emitULEB128(DecoderIndex);

  emitComment(Indent) << "decode to " << Encoding.getName() << " using decoder "
                      << DecoderIndex << '\n';
}

void DecoderTableEmitter::emitNode(const DecoderTreeNode *N, indent Indent) {
  switch (N->getKind()) {
  case DecoderTreeNode::CheckAny:
    return emitCheckAnyNode(static_cast<const CheckAnyNode *>(N), Indent);
  case DecoderTreeNode::CheckAll:
    return emitCheckAllNode(static_cast<const CheckAllNode *>(N), Indent);
  case DecoderTreeNode::SwitchField:
    return emitSwitchFieldNode(static_cast<const SwitchFieldNode *>(N), Indent);
  case DecoderTreeNode::CheckField:
    return emitCheckFieldNode(static_cast<const CheckFieldNode *>(N), Indent);
  case DecoderTreeNode::CheckPredicate:
    return emitCheckPredicateNode(static_cast<const CheckPredicateNode *>(N),
                                  Indent);
  case DecoderTreeNode::SoftFail:
    return emitSoftFailNode(static_cast<const SoftFailNode *>(N), Indent);
  case DecoderTreeNode::Decode:
    return emitDecodeNode(static_cast<const DecodeNode *>(N), Indent);
  }
  llvm_unreachable("Unknown node kind");
}

void DecoderTableEmitter::emitTable(StringRef TableName, unsigned BitWidth,
                                    const DecoderTreeNode *Root) {
  analyzeNode(Root);

  unsigned TableSize = computeTableSize(Root, BitWidth);
  OS << "static const uint8_t " << TableName << "[" << TableSize << "] = {\n";

  // Calculate the number of decimal places for table indices.
  // This is simply log10 of the table size.
  IndexWidth = 1;
  for (unsigned S = TableSize; S /= 10;)
    ++IndexWidth;

  CurrentIndex = 0;

  // When specializing decoders per bit width, each decoder table will begin
  // with the bitwidth for that table.
  if (BitWidth) {
    emitStartLine();
    emitULEB128(BitWidth);
    emitComment(indent(0)) << "BitWidth " << BitWidth << '\n';
  }

  emitNode(Root, indent(0));
  assert(CurrentIndex == TableSize &&
         "The size of the emitted table differs from the calculated one");

  OS << "};\n";
}
