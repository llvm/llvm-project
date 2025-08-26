//===---------------- DecoderEmitter.cpp - Decoder Generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// It contains the tablegen backend that emits the decoder functions for
// targets with fixed/variable length instruction set.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenHwModes.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenTarget.h"
#include "Common/InfoByHwMode.h"
#include "Common/VarLenCodeEmitterGen.h"
#include "TableGenBackends.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "decoder-emitter"

extern cl::OptionCategory DisassemblerEmitterCat;

enum SuppressLevel {
  SUPPRESSION_DISABLE,
  SUPPRESSION_LEVEL1,
  SUPPRESSION_LEVEL2
};

static cl::opt<SuppressLevel> DecoderEmitterSuppressDuplicates(
    "suppress-per-hwmode-duplicates",
    cl::desc("Suppress duplication of instrs into per-HwMode decoder tables"),
    cl::values(
        clEnumValN(
            SUPPRESSION_DISABLE, "O0",
            "Do not prevent DecoderTable duplications caused by HwModes"),
        clEnumValN(
            SUPPRESSION_LEVEL1, "O1",
            "Remove duplicate DecoderTable entries generated due to HwModes"),
        clEnumValN(
            SUPPRESSION_LEVEL2, "O2",
            "Extract HwModes-specific instructions into new DecoderTables, "
            "significantly reducing Table Duplications")),
    cl::init(SUPPRESSION_DISABLE), cl::cat(DisassemblerEmitterCat));

static cl::opt<bool> UseFnTableInDecodeToMCInst(
    "use-fn-table-in-decode-to-mcinst",
    cl::desc(
        "Use a table of function pointers instead of a switch case in the\n"
        "generated `decodeToMCInst` function. Helps improve compile time\n"
        "of the generated code."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

// Enabling this option requires use of different `InsnType` for different
// bitwidths and defining `InsnBitWidth` template specialization for the
// `InsnType` types used. Some common specializations are already defined in
// MCDecoder.h.
static cl::opt<bool> SpecializeDecodersPerBitwidth(
    "specialize-decoders-per-bitwidth",
    cl::desc("Specialize the generated `decodeToMCInst` function per bitwidth. "
             "Helps reduce the code size."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

static cl::opt<bool> IgnoreNonDecodableOperands(
    "ignore-non-decodable-operands",
    cl::desc(
        "Do not issue an error if an operand cannot be decoded automatically."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

static cl::opt<bool> IgnoreFullyDefinedOperands(
    "ignore-fully-defined-operands",
    cl::desc(
        "Do not automatically decode operands with no '?' in their encoding."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

STATISTIC(NumEncodings, "Number of encodings considered");
STATISTIC(NumEncodingsLackingDisasm,
          "Number of encodings without disassembler info");
STATISTIC(NumInstructions, "Number of instructions considered");
STATISTIC(NumEncodingsSupported, "Number of encodings supported");
STATISTIC(NumEncodingsOmitted, "Number of encodings omitted");

/// Similar to KnownBits::print(), but allows you to specify a character to use
/// to print unknown bits.
static void printKnownBits(raw_ostream &OS, const KnownBits &Bits,
                           char Unknown) {
  for (unsigned I = Bits.getBitWidth(); I--;) {
    if (Bits.Zero[I] && Bits.One[I])
      OS << '!';
    else if (Bits.Zero[I])
      OS << '0';
    else if (Bits.One[I])
      OS << '1';
    else
      OS << Unknown;
  }
}

namespace {

// Represents a span of bits in the instruction encoding that's based on a span
// of bits in an operand's encoding.
//
// Width is the width of the span.
// Base is the starting position of that span in the instruction encoding.
// Offset if the starting position of that span in the operand's encoding.
// That is, bits {Base + Width - 1, Base} in the instruction encoding form
// bits {Offset + Width - 1, Offset} in the operands encoding.
struct EncodingField {
  unsigned Base, Width, Offset;
  EncodingField(unsigned B, unsigned W, unsigned O)
      : Base(B), Width(W), Offset(O) {}
};

struct OperandInfo {
  std::vector<EncodingField> Fields;
  std::string Decoder;
  bool HasCompleteDecoder;
  std::optional<uint64_t> InitValue;

  OperandInfo(std::string D, bool HCD) : Decoder(D), HasCompleteDecoder(HCD) {}

  void addField(unsigned Base, unsigned Width, unsigned Offset) {
    Fields.emplace_back(Base, Width, Offset);
  }

  unsigned numFields() const { return Fields.size(); }

  ArrayRef<EncodingField> fields() const { return Fields; }
};

/// Represents a parsed InstructionEncoding record or a record derived from it.
class InstructionEncoding {
  /// The Record this encoding originates from.
  const Record *EncodingDef;

  /// The instruction this encoding is for.
  const CodeGenInstruction *Inst;

  /// The name of this encoding (for debugging purposes).
  std::string Name;

  /// The namespace in which this encoding exists.
  StringRef DecoderNamespace;

  /// Known bits of this encoding. This is the value of the `Inst` field
  /// with any variable references replaced with '?'.
  KnownBits InstBits;

  /// Mask of bits that should be considered unknown during decoding.
  /// This is the value of the `SoftFail` field.
  APInt SoftFailMask;

  /// The name of the function to use for decoding. May be an empty string,
  /// meaning the decoder is generated.
  StringRef DecoderMethod;

  /// Whether the custom decoding function always succeeds. If a custom decoder
  /// function is specified, the value is taken from the target description,
  /// otherwise it is inferred.
  bool HasCompleteDecoder;

  /// Information about the operands' contribution to this encoding.
  SmallVector<OperandInfo, 16> Operands;

public:
  InstructionEncoding(const Record *EncodingDef,
                      const CodeGenInstruction *Inst);

  /// Returns the Record this encoding originates from.
  const Record *getRecord() const { return EncodingDef; }

  /// Returns the instruction this encoding is for.
  const CodeGenInstruction *getInstruction() const { return Inst; }

  /// Returns the name of this encoding, for debugging purposes.
  StringRef getName() const { return Name; }

  /// Returns the namespace in which this encoding exists.
  StringRef getDecoderNamespace() const { return DecoderNamespace; }

  /// Returns the size of this encoding, in bits.
  unsigned getBitWidth() const { return InstBits.getBitWidth(); }

  /// Returns the known bits of this encoding.
  const KnownBits &getInstBits() const { return InstBits; }

  /// Returns a mask of bits that should be considered unknown during decoding.
  const APInt &getSoftFailMask() const { return SoftFailMask; }

  /// Returns the known bits of this encoding that must match for
  /// successful decoding.
  KnownBits getMandatoryBits() const {
    KnownBits EncodingBits = InstBits;
    // Mark all bits that are allowed to change according to SoftFail mask
    // as unknown.
    EncodingBits.Zero &= ~SoftFailMask;
    EncodingBits.One &= ~SoftFailMask;
    return EncodingBits;
  }

  /// Returns the name of the function to use for decoding, or an empty string
  /// if the decoder is generated.
  StringRef getDecoderMethod() const { return DecoderMethod; }

  /// Returns whether the decoder (either generated or specified by the user)
  /// always succeeds.
  bool hasCompleteDecoder() const { return HasCompleteDecoder; }

  /// Returns information about the operands' contribution to this encoding.
  ArrayRef<OperandInfo> getOperands() const { return Operands; }

private:
  void parseVarLenEncoding(const VarLenInst &VLI);
  void parseFixedLenEncoding(const BitsInit &RecordInstBits);

  void parseVarLenOperands(const VarLenInst &VLI);
  void parseFixedLenOperands(const BitsInit &Bits);
};

/// Sorting predicate to sort encoding IDs by encoding width.
class LessEncodingIDByWidth {
  ArrayRef<InstructionEncoding> Encodings;

public:
  explicit LessEncodingIDByWidth(ArrayRef<InstructionEncoding> Encodings)
      : Encodings(Encodings) {}

  bool operator()(unsigned ID1, unsigned ID2) const {
    return Encodings[ID1].getBitWidth() < Encodings[ID2].getBitWidth();
  }
};

class DecoderTreeNode {
public:
  virtual ~DecoderTreeNode() = default;

  enum KindTy {
    AnyOf,
    AllOf,
    CheckField,
    SwitchField,
    CheckPredicate,
    SoftFail,
    Decode,
  };

  KindTy getKind() const { return Kind; }

protected:
  explicit DecoderTreeNode(KindTy Kind) : Kind(Kind) {}

private:
  KindTy Kind;
};

// Child of AnyOf or SwitchField.
class AllOfNode : public DecoderTreeNode {
  SmallVector<std::unique_ptr<DecoderTreeNode>, 0> Children;

  static const DecoderTreeNode *
  mapElement(decltype(Children)::const_reference Element) {
    return Element.get();
  }

public:
  AllOfNode() : DecoderTreeNode(AllOf) {}

  void addChild(std::unique_ptr<DecoderTreeNode> N) {
    Children.push_back(std::move(N));
  }

  using child_iterator = mapped_iterator<decltype(Children)::const_iterator,
                                         decltype(&mapElement)>;

  child_iterator child_begin() const {
    return child_iterator(Children.begin(), mapElement);
  }

  child_iterator child_end() const {
    return child_iterator(Children.end(), mapElement);
  }

  iterator_range<child_iterator> children() const {
    return make_range(child_begin(), child_end());
  }
};

// Either root, follows CheckField in AllOf, or a child of SwitchField.
// Children are only:
//   Terminal (AllOf CheckField -> ... -> Decode)
//   (AllOf CheckField -> AnyOf)
//   SwitchField
// If a child fails, proceed to next child.
// If last child fails, AnyOf fails.
class AnyOfNode : public DecoderTreeNode {
  SmallVector<std::unique_ptr<DecoderTreeNode>, 0> Children;

  static const DecoderTreeNode *
  mapElement(decltype(Children)::const_reference Element) {
    return Element.get();
  }

public:
  AnyOfNode() : DecoderTreeNode(AnyOf) {}

  void addChild(std::unique_ptr<DecoderTreeNode> N) {
    Children.push_back(std::move(N));
  }

  using child_iterator = mapped_iterator<decltype(Children)::const_iterator,
                                         decltype(&mapElement)>;

  child_iterator child_begin() const {
    return child_iterator(Children.begin(), mapElement);
  }

  child_iterator child_end() const {
    return child_iterator(Children.end(), mapElement);
  }

  iterator_range<child_iterator> children() const {
    return make_range(child_begin(), child_end());
  }
};

// Always a child of AnyOf.
// Equivalent to:
// AnyOf
//   `CheckField
//   |  `...
//   `CheckField
//      `...
class SwitchFieldNode : public DecoderTreeNode {
  unsigned StartBit;
  unsigned NumBits;
  std::map<uint64_t, std::unique_ptr<DecoderTreeNode>> Cases;

  static std::pair<uint64_t, const DecoderTreeNode *>
  mapElement(decltype(Cases)::const_reference Element) {
    return std::pair(Element.first, Element.second.get());
  }

public:
  SwitchFieldNode(unsigned StartBit, unsigned NumBits)
      : DecoderTreeNode(SwitchField), StartBit(StartBit), NumBits(NumBits) {}

  void addCase(uint64_t Value, std::unique_ptr<DecoderTreeNode> N) {
    Cases.try_emplace(Value, std::move(N));
  }

  unsigned getStartBit() const { return StartBit; }

  unsigned getNumBits() const { return NumBits; }

  using case_iterator =
      mapped_iterator<decltype(Cases)::const_iterator, decltype(&mapElement)>;

  case_iterator case_begin() const {
    return case_iterator(Cases.begin(), mapElement);
  }

  case_iterator case_end() const {
    return case_iterator(Cases.end(), mapElement);
  }

  iterator_range<case_iterator> cases() const {
    return make_range(case_begin(), case_end());
  }
};

// Always a child of AllOf.
class CheckFieldNode : public DecoderTreeNode {
  unsigned StartBit;
  unsigned NumBits;
  uint64_t Value;

public:
  CheckFieldNode(unsigned StartBit, unsigned NumBits, uint64_t Value)
      : DecoderTreeNode(CheckField), StartBit(StartBit), NumBits(NumBits),
        Value(Value) {}

  unsigned getStartBit() const { return StartBit; }

  unsigned getNumBits() const { return NumBits; }

  uint64_t getValue() const { return Value; }
};

// Always a child of AllOf.
class CheckPredicateNode : public DecoderTreeNode {
  unsigned Index;

public:
  explicit CheckPredicateNode(unsigned Index)
      : DecoderTreeNode(CheckPredicate), Index(Index) {}

  unsigned getPredicateIndex() const { return Index; }
};

// Always a child of AllOf.
class SoftFailNode : public DecoderTreeNode {
  uint64_t PositiveMask, NegativeMask;

public:
  explicit SoftFailNode(uint64_t PositiveMask, uint64_t NegativeMask)
      : DecoderTreeNode(SoftFail), PositiveMask(PositiveMask),
        NegativeMask(NegativeMask) {}

  uint64_t getPositiveMask() const { return PositiveMask; }
  uint64_t getNegativeMask() const { return NegativeMask; }
};

// Always a child of AllOf.
class DecodeNode : public DecoderTreeNode {
  unsigned EncodingID;
  unsigned Index;

public:
  DecodeNode(unsigned EncodingID, unsigned Index)
      : DecoderTreeNode(Decode), EncodingID(EncodingID), Index(Index) {}

  unsigned getEncodingID() const { return EncodingID; }

  unsigned getDecoderIndex() const { return Index; }
};

typedef SmallSetVector<CachedHashString, 16> PredicateSet;
typedef SmallSetVector<CachedHashString, 16> DecoderSet;

struct DecoderTableInfo {
  PredicateSet Predicates;
  DecoderSet Decoders;
};

using NamespacesHwModesMap = std::map<StringRef, std::set<unsigned>>;

class DecoderEmitter {
  const RecordKeeper &RK;
  CodeGenTarget Target;
  const CodeGenHwModes &CGH;

  /// All parsed encodings.
  std::vector<InstructionEncoding> Encodings;

  /// Encodings IDs for each HwMode. An ID is an index into Encodings.
  SmallDenseMap<unsigned, std::vector<unsigned>> EncodingIDsByHwMode;

public:
  explicit DecoderEmitter(const RecordKeeper &RK);

  const CodeGenTarget &getTarget() const { return Target; }

  // Emit the decoder state machine table.
  void emitDecoderTable(formatted_raw_ostream &OS, const DecoderTreeNode *Tree,
                        StringRef Namespace, unsigned HwModeID,
                        unsigned BitWidth) const;
  void emitInstrLenTable(formatted_raw_ostream &OS,
                         ArrayRef<unsigned> InstrLen) const;
  void emitPredicateFunction(formatted_raw_ostream &OS,
                             PredicateSet &Predicates) const;
  void emitDecoderFunction(formatted_raw_ostream &OS,
                           const DecoderSet &Decoders,
                           unsigned BucketBitWidth) const;

  // run - Output the code emitter
  void run(raw_ostream &o) const;

private:
  void collectHwModesReferencedForEncodings(
      std::vector<unsigned> &HwModeIDs,
      NamespacesHwModesMap &NamespacesWithHwModes) const;

  void
  handleHwModesUnrelatedEncodings(unsigned EncodingID,
                                  ArrayRef<unsigned> HwModeIDs,
                                  NamespacesHwModesMap &NamespacesWithHwModes);

  void parseInstructionEncodings();
};

} // end anonymous namespace

namespace {

/// Filter - Filter works with FilterChooser to produce the decoding tree for
/// the ISA.
///
/// It is useful to think of a Filter as governing the switch stmts of the
/// decoding tree in a certain level.  Each case stmt delegates to an inferior
/// FilterChooser to decide what further decoding logic to employ, or in another
/// words, what other remaining bits to look at.  The FilterChooser eventually
/// chooses a best Filter to do its job.
///
/// This recursive scheme ends when the number of Opcodes assigned to the
/// FilterChooser becomes 1 or if there is a conflict.  A conflict happens when
/// the Filter/FilterChooser combo does not know how to distinguish among the
/// Opcodes assigned.
///
/// An example of a conflict is
///
/// Decoding Conflict:
///     ................................
///     1111............................
///     1111010.........................
///     1111010...00....................
///     1111010...00........0001........
///     111101000.00........0001........
///     111101000.00........00010000....
///     111101000_00________00010000____  VST4q8a
///     111101000_00________00010000____  VST4q8b
///
/// The Debug output shows the path that the decoding tree follows to reach the
/// the conclusion that there is a conflict.  VST4q8a is a vst4 to double-spaced
/// even registers, while VST4q8b is a vst4 to double-spaced odd registers.
///
/// The encoding info in the .td files does not specify this meta information,
/// which could have been used by the decoder to resolve the conflict.  The
/// decoder could try to decode the even/odd register numbering and assign to
/// VST4q8a or VST4q8b, but for the time being, the decoder chooses the "a"
/// version and return the Opcode since the two have the same Asm format string.
struct Filter {
  unsigned StartBit; // the starting bit position
  unsigned NumBits;  // number of bits to filter

  // Map of well-known segment value to the set of uid's with that value.
  std::map<uint64_t, std::vector<unsigned>> FilteredIDs;

  // Set of uid's with non-constant segment values.
  std::vector<unsigned> VariableIDs;

  Filter(ArrayRef<InstructionEncoding> Encodings,
         ArrayRef<unsigned> EncodingIDs, unsigned StartBit, unsigned NumBits);

  // Returns the number of fanout produced by the filter.  More fanout implies
  // the filter distinguishes more categories of instructions.
  unsigned usefulness() const;
}; // end class Filter

// These are states of our finite state machines used in FilterChooser's
// filterProcessor() which produces the filter candidates to use.
enum bitAttr_t {
  ATTR_NONE,
  ATTR_FILTERED,
  ATTR_ALL_SET,
  ATTR_ALL_UNSET,
  ATTR_MIXED
};

/// FilterChooser - FilterChooser chooses the best filter among a set of Filters
/// in order to perform the decoding of instructions at the current level.
///
/// Decoding proceeds from the top down.  Based on the well-known encoding bits
/// of instructions available, FilterChooser builds up the possible Filters that
/// can further the task of decoding by distinguishing among the remaining
/// candidate instructions.
///
/// Once a filter has been chosen, it is called upon to divide the decoding task
/// into sub-tasks and delegates them to its inferior FilterChoosers for further
/// processings.
///
/// It is useful to think of a Filter as governing the switch stmts of the
/// decoding tree.  And each case is delegated to an inferior FilterChooser to
/// decide what further remaining bits to look at.

class DecoderTreeBuilder {
  const CodeGenTarget &Target;
  ArrayRef<InstructionEncoding> Encodings;
  DecoderTableInfo &TableInfo;

  struct Island {
    unsigned StartBit;
    unsigned NumBits;
    uint64_t FieldVal;
  };

  // Calculates the island(s) needed to decode the instruction.
  // This returns a list of undecoded bits of an instructions, for example,
  // Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
  // decoded bits in order to verify that the instruction matches the Opcode.
  static std::vector<Island> getIslands(const KnownBits &EncodingBits,
                                        const KnownBits &FilterBits);

public:
  DecoderTreeBuilder(const CodeGenTarget &Target,
                     ArrayRef<InstructionEncoding> Encodings,
                     DecoderTableInfo &TableInfo)
      : Target(Target), Encodings(Encodings), TableInfo(TableInfo) {}

  std::unique_ptr<DecoderTreeNode> buildTree(ArrayRef<unsigned> EncodingIDs) {
    SmallVector<unsigned, 0> UnfilteredIDs(EncodingIDs);
    stable_sort(UnfilteredIDs, LessEncodingIDByWidth(Encodings));
    // AnyOf node with AllOf/SwitchField children.
    return buildAnyOfNode(UnfilteredIDs);
  }

  DecoderTreeBuilder(const DecoderTreeBuilder &) = delete;
  void operator=(const DecoderTreeBuilder &) = delete;

private:
  static void emitBinaryParser(raw_ostream &OS, indent Indent,
                               const OperandInfo &OpInfo);
  static void emitDecoder(raw_ostream &OS, indent Indent,
                          const InstructionEncoding &Encoding);
  unsigned getDecoderIndex(const InstructionEncoding &Encoding);

  static bool emitPredicateMatchAux(StringRef PredicateNamespace,
                                    const Init &Val, bool ParenIfBinOp,
                                    raw_ostream &OS);
  static bool emitPredicateMatch(StringRef PredicateNamespace, raw_ostream &OS,
                                 const InstructionEncoding &Encoding);
  unsigned getPredicateIndex(const InstructionEncoding &Encoding) const;

  /// dumpStack - dumpStack traverses the filter chooser chain and calls
  /// dumpFilterArray on each filter chooser up to the top level one.
  void dumpStack(raw_ostream &OS, indent Indent, unsigned PadToWidth,
                 const KnownBits &FilterBits) const;

  // reportRegion is a helper function for filterProcessor to mark a region as
  // eligible for use as a filter region.
  void reportRegion(ArrayRef<unsigned> EncodingIDs,
                    std::vector<std::unique_ptr<Filter>> &Filters, bitAttr_t RA,
                    unsigned StartBit, unsigned BitIndex,
                    bool AllowMixed) const;

  /// Scans the well-known encoding bits of the encodings and, builds up a list
  /// of candidate filters, and then returns the best one, if any.
  std::unique_ptr<Filter> findBestFilter(ArrayRef<unsigned> EncodingIDs,
                                         const KnownBits &FilterBits,
                                         ArrayRef<bitAttr_t> BitAttrs,
                                         bool AllowMixed,
                                         bool Greedy = true) const;

  std::unique_ptr<Filter> findBestFilter(ArrayRef<unsigned> EncodingIDs,
                                         const KnownBits &FilterBits) const;

  std::unique_ptr<DecoderTreeNode>
  buildTerminalNode(unsigned EncodingID, const KnownBits &FilterBits);

  std::unique_ptr<DecoderTreeNode> buildAllOfOrSwitchNode(
      const std::map<uint64_t, std::vector<unsigned>> &FilteredIDs,
      unsigned StartBit, unsigned NumBits, const KnownBits &FilterBits);

  std::unique_ptr<DecoderTreeNode>
  buildAnyOfNode(ArrayRef<unsigned> EncodingIDs,
                 KnownBits FilterBits = KnownBits());

public:
  void dump(ArrayRef<unsigned> EncodingIDs, const KnownBits &FilterBits) const;
};

class DecoderTableEmitter {
  const CodeGenTarget &Target;
  ArrayRef<InstructionEncoding> Encodings;
  formatted_raw_ostream OS;
  unsigned IndexWidth;
  unsigned CurrentIndex;
  unsigned CommentIndex;
  bool HasCheckPredicate = false;
  bool HasSoftFail = false;
  bool HasTryDecode = false;

public:
  DecoderTableEmitter(const CodeGenTarget &Target,
                      ArrayRef<InstructionEncoding> Encodings, raw_ostream &OS)
      : Target(Target), Encodings(Encodings), OS(OS) {}

  void emitTable(StringRef TableName, unsigned BitWidth,
                 const DecoderTreeNode *Root);

  bool hasCheckPredicate() const { return HasCheckPredicate; }

  bool hasSoftFail() const { return HasSoftFail; }

  bool hasTryDecode() const { return HasTryDecode; }

private:
  unsigned computeNodeSize(const DecoderTreeNode *N) const;

  unsigned computeTableSize(unsigned BitWidth,
                            const DecoderTreeNode *Root) const {
    unsigned Size = 0;
    if (SpecializeDecodersPerBitwidth)
      Size = getULEB128Size(BitWidth);
    Size += computeNodeSize(Root);
    return Size;
  }

  void emitStartLine() {
    CommentIndex = CurrentIndex;
    OS.indent(2);
  }

  void emitOpcode(StringRef Name) {
    emitStartLine();
    OS << "MCD::" << Name << ", ";
    ++CurrentIndex;
  }

  void emitByte(uint8_t Val) {
    OS << static_cast<unsigned>(Val) << ", ";
    ++CurrentIndex;
  }

  void emitUInt8(unsigned Val) {
    assert(isUInt<8>(Val));
    emitByte(Val);
  }

  void emitULEB128(uint64_t Val) {
    while (Val >= 0x80) {
      emitByte((Val & 0x7F) | 0x80);
      Val >>= 7;
    }
    emitByte(Val);
  }

  formatted_raw_ostream &emitComment(indent Indent) {
    constexpr unsigned CommentColumn = 45;
    if (OS.getColumn() > CommentColumn)
      OS << '\n';
    OS.PadToColumn(CommentColumn);
    OS << "// " << format_decimal(CommentIndex, IndexWidth) << ": " << Indent;
    return OS;
  }

  void emitAnyOfNode(const AnyOfNode *N, indent Indent);
  void emitAllOfNode(const AllOfNode *N, indent Indent);
  void emitSwitchFieldNode(const SwitchFieldNode *N, indent Indent);
  void emitCheckFieldNode(const CheckFieldNode *N, indent Indent);
  void emitCheckPredicateNode(const CheckPredicateNode *N, indent Indent);
  void emitSoftFailNode(const SoftFailNode *N, indent Indent);
  void emitDecodeNode(const DecodeNode *N, indent Indent);
  void emitNode(const DecoderTreeNode *N, indent Indent);
};

} // end anonymous namespace

///////////////////////////
//                       //
// Filter Implementation //
//                       //
///////////////////////////

Filter::Filter(ArrayRef<InstructionEncoding> Encodings,
               ArrayRef<unsigned> EncodingIDs, unsigned StartBit,
               unsigned NumBits)
    : StartBit(StartBit), NumBits(NumBits) {
  for (unsigned EncodingID : EncodingIDs) {
    const InstructionEncoding &Encoding = Encodings[EncodingID];
    KnownBits EncodingBits = Encoding.getMandatoryBits();

    // Scans the segment for possibly well-specified encoding bits.
    KnownBits FieldBits = EncodingBits.extractBits(NumBits, StartBit);

    if (FieldBits.isConstant()) {
      // The encoding bits are well-known.  Lets add the uid of the
      // instruction into the bucket keyed off the constant field value.
      FilteredIDs[FieldBits.getConstant().getZExtValue()].push_back(EncodingID);
    } else {
      // Some of the encoding bit(s) are unspecified.  This contributes to
      // one additional member of "Variable" instructions.
      VariableIDs.push_back(EncodingID);
    }
  }

  assert((FilteredIDs.size() + VariableIDs.size() > 0) &&
         "Filter returns no instruction categories");
}

// Returns the number of fanout produced by the filter.  More fanout implies
// the filter distinguishes more categories of instructions.
unsigned Filter::usefulness() const {
  return FilteredIDs.size() + VariableIDs.empty();
}

static bool doesOpcodeNeedPredicate(const InstructionEncoding &Encoding);

std::unique_ptr<DecoderTreeNode>
DecoderTreeBuilder::buildTerminalNode(unsigned EncodingID,
                                      const KnownBits &FilterBits) {
  const InstructionEncoding &Encoding = Encodings[EncodingID];
  auto N = std::make_unique<AllOfNode>();

  if (doesOpcodeNeedPredicate(Encoding)) {
    unsigned PredicateIndex = getPredicateIndex(Encoding);
    N->addChild(std::make_unique<CheckPredicateNode>(PredicateIndex));
  }

  std::vector<Island> Islands =
      getIslands(Encoding.getMandatoryBits(), FilterBits);
  for (const Island &Ilnd : reverse(Islands)) {
    N->addChild(std::make_unique<CheckFieldNode>(Ilnd.StartBit, Ilnd.NumBits,
                                                 Ilnd.FieldVal));
  }

  const KnownBits &InstBits = Encoding.getInstBits();
  const APInt &SoftFailMask = Encoding.getSoftFailMask();
  if (!SoftFailMask.isZero()) {
    APInt PositiveMask = InstBits.Zero & SoftFailMask;
    APInt NegativeMask = InstBits.One & SoftFailMask;
    N->addChild(std::make_unique<SoftFailNode>(PositiveMask.getZExtValue(),
                                               NegativeMask.getZExtValue()));
  }

  unsigned DecoderIndex = getDecoderIndex(Encoding);
  N->addChild(std::make_unique<DecodeNode>(EncodingID, DecoderIndex));

  return N;
}

// Makes (AllOf CheckField, AnyOf) or SwitchField.
std::unique_ptr<DecoderTreeNode> DecoderTreeBuilder::buildAllOfOrSwitchNode(
    const std::map<uint64_t, std::vector<unsigned>> &FilteredIDs,
    unsigned StartBit, unsigned NumBits, const KnownBits &FilterBits) {
  // Optimize one-case SwitchField to CheckField.
  if (FilteredIDs.size() == 1) {
    const auto &[FilterVal, InferiorEncodingIDs] = *FilteredIDs.begin();
    // Create a new filter by inserting the field bits into the parent filter.
    APInt FieldBits(NumBits, FilterVal);
    KnownBits InferiorFilterBits = FilterBits;
    InferiorFilterBits.insertBits(KnownBits::makeConstant(FieldBits), StartBit);

    auto N = std::make_unique<AllOfNode>();
    N->addChild(std::make_unique<CheckFieldNode>(StartBit, NumBits, FilterVal));
    // AnyOf child with AllOf/SwitchField children.
    N->addChild(buildAnyOfNode(InferiorEncodingIDs, InferiorFilterBits));
    return N;
  }

  auto N = std::make_unique<SwitchFieldNode>(StartBit, NumBits);
  for (const auto &[FilterVal, InferiorEncodingIDs] : FilteredIDs) {
    // Create a new filter by inserting the field bits into the parent filter.
    APInt FieldBits(NumBits, FilterVal);
    KnownBits InferiorFilterBits = FilterBits;
    InferiorFilterBits.insertBits(KnownBits::makeConstant(FieldBits), StartBit);
    // AnyOf child with AllOf/SwitchField children.
    N->addCase(FilterVal,
               buildAnyOfNode(InferiorEncodingIDs, InferiorFilterBits));
  }
  return N;
}

// AnyOf with children Terminal, (AllOf CheckField, AnyOf) or SwitchField.
std::unique_ptr<DecoderTreeNode>
DecoderTreeBuilder::buildAnyOfNode(ArrayRef<unsigned> EncodingIDs,
                                   KnownBits FilterBits) {
  SmallVector<unsigned, 0> UnfilteredIDs(EncodingIDs);

  // Children are only Terminal, (AllOf CheckField, AnyOf) or SwitchField.
  auto N = std::make_unique<AnyOfNode>();

  while (!UnfilteredIDs.empty()) {
    // Filter width is the width of the smallest encoding.
    unsigned FilterWidth = Encodings[UnfilteredIDs.front()].getBitWidth();
    FilterBits = FilterBits.anyext(FilterWidth);

    if (UnfilteredIDs.size() == 1) {
      N->addChild(buildTerminalNode(UnfilteredIDs.front(), FilterBits));
      break;
    }

    if (std::unique_ptr<Filter> F = findBestFilter(UnfilteredIDs, FilterBits)) {
      // (AllOf CheckField, AnyOf) or SwitchField child.
      N->addChild(buildAllOfOrSwitchNode(F->FilteredIDs, F->StartBit,
                                         F->NumBits, FilterBits));
      UnfilteredIDs.assign(F->VariableIDs.begin(), F->VariableIDs.end());
      continue;
    }

    // Print out useful conflict information for postmortem analysis.
    errs() << "Decoding Conflict:\n";
    dump(UnfilteredIDs, FilterBits);
    PrintFatalError("Decoding conflict encountered");
    return nullptr;
  }

  return N;
}

// Emit the decoder state machine table.
void DecoderEmitter::emitDecoderTable(formatted_raw_ostream &OS,
                                      const DecoderTreeNode *Tree,
                                      StringRef Namespace, unsigned HwModeID,
                                      unsigned BitWidth) const {
  SmallString<32> TableName("DecoderTable");
  TableName.append(Namespace);
  if (HwModeID != DefaultMode)
    TableName.append({"_", Target.getHwModes().getModeName(HwModeID)});
  TableName.append(std::to_string(BitWidth));

  DecoderTableEmitter TableEmitter(Target, Encodings, OS);
  TableEmitter.emitTable(TableName, BitWidth, Tree);
}

void DecoderEmitter::emitInstrLenTable(formatted_raw_ostream &OS,
                                       ArrayRef<unsigned> InstrLen) const {
  OS << "static const uint8_t InstrLenTable[] = {\n";
  for (unsigned Len : InstrLen)
    OS << Len << ",\n";
  OS << "};\n\n";
}

void DecoderEmitter::emitPredicateFunction(formatted_raw_ostream &OS,
                                           PredicateSet &Predicates) const {
  // The predicate function is just a big switch statement based on the
  // input predicate index.
  OS << "static bool checkDecoderPredicate(unsigned Idx, const FeatureBitset "
        "&Bits) {\n";
  OS << "  switch (Idx) {\n";
  OS << "  default: llvm_unreachable(\"Invalid index!\");\n";
  for (const auto &[Index, Predicate] : enumerate(Predicates)) {
    OS << "  case " << Index << ":\n";
    OS << "    return (" << Predicate << ");\n";
  }
  OS << "  }\n";
  OS << "}\n\n";
}

void DecoderEmitter::emitDecoderFunction(formatted_raw_ostream &OS,
                                         const DecoderSet &Decoders,
                                         unsigned BucketBitWidth) const {
  // The decoder function is just a big switch statement or a table of function
  // pointers based on the input decoder index.

  // TODO: When InsnType is large, using uint64_t limits all fields to 64 bits
  // It would be better for emitBinaryParser to use a 64-bit tmp whenever
  // possible but fall back to an InsnType-sized tmp for truly large fields.
  StringRef TmpTypeDecl =
      "using TmpType = std::conditional_t<std::is_integral<InsnType>::value, "
      "InsnType, uint64_t>;\n";
  StringRef DecodeParams =
      "DecodeStatus S, InsnType insn, MCInst &MI, uint64_t Address, const "
      "MCDisassembler *Decoder, bool &DecodeComplete";

  // Print the name of the decode function to OS.
  auto PrintDecodeFnName = [&OS, BucketBitWidth](unsigned DecodeIdx) {
    OS << "decodeFn";
    if (BucketBitWidth != 0) {
      OS << '_' << BucketBitWidth << "bit";
    }
    OS << '_' << DecodeIdx;
  };

  // Print the template statement.
  auto PrintTemplate = [&OS, BucketBitWidth]() {
    OS << "template <typename InsnType>\n";
    OS << "static ";
    if (BucketBitWidth != 0)
      OS << "std::enable_if_t<InsnBitWidth<InsnType> == " << BucketBitWidth
         << ", DecodeStatus>\n";
    else
      OS << "DecodeStatus ";
  };

  if (UseFnTableInDecodeToMCInst) {
    // Emit a function for each case first.
    for (const auto &[Index, Decoder] : enumerate(Decoders)) {
      PrintTemplate();
      PrintDecodeFnName(Index);
      OS << "(" << DecodeParams << ") {\n";
      OS << "  using namespace llvm::MCD;\n";
      OS << "  " << TmpTypeDecl;
      OS << "  [[maybe_unused]] TmpType tmp;\n";
      OS << Decoder;
      OS << "  return S;\n";
      OS << "}\n\n";
    }
  }

  OS << "// Handling " << Decoders.size() << " cases.\n";
  PrintTemplate();
  OS << "decodeToMCInst(unsigned Idx, " << DecodeParams << ") {\n";
  OS << "  using namespace llvm::MCD;\n";
  OS << "  DecodeComplete = true;\n";

  if (UseFnTableInDecodeToMCInst) {
    // Build a table of function pointers
    OS << "  using DecodeFnTy = DecodeStatus (*)(" << DecodeParams << ");\n";
    OS << "  static constexpr DecodeFnTy decodeFnTable[] = {\n";
    for (size_t Index : llvm::seq(Decoders.size())) {
      OS << "    ";
      PrintDecodeFnName(Index);
      OS << ",\n";
    }
    OS << "  };\n";
    OS << "  if (Idx >= " << Decoders.size() << ")\n";
    OS << "    llvm_unreachable(\"Invalid decoder index!\");\n";
    OS << "  return decodeFnTable[Idx](S, insn, MI, Address, Decoder, "
          "DecodeComplete);\n";
  } else {
    OS << "  " << TmpTypeDecl;
    OS << "  TmpType tmp;\n";
    OS << "  switch (Idx) {\n";
    OS << "  default: llvm_unreachable(\"Invalid decoder index!\");\n";
    for (const auto &[Index, Decoder] : enumerate(Decoders)) {
      OS << "  case " << Index << ":\n";
      OS << Decoder;
      OS << "    return S;\n";
    }
    OS << "  }\n";
  }
  OS << "}\n";
}

/// dumpStack - dumpStack traverses the filter chooser chain and calls
/// dumpFilterArray on each filter chooser up to the top level one.
void DecoderTreeBuilder::dumpStack(raw_ostream &OS, indent Indent,
                                   unsigned PadToWidth,
                                   const KnownBits &FilterBits) const {
  assert(PadToWidth >= FilterBits.getBitWidth());
  OS << Indent << indent(PadToWidth - FilterBits.getBitWidth());
  printKnownBits(OS, FilterBits, '.');
  OS << '\n';
}

// Calculates the island(s) needed to decode the instruction.
// This returns a list of undecoded bits of an instructions, for example,
// Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
// decoded bits in order to verify that the instruction matches the Opcode.
std::vector<DecoderTreeBuilder::Island>
DecoderTreeBuilder::getIslands(const KnownBits &EncodingBits,
                               const KnownBits &FilterBits) {
  std::vector<Island> Islands;
  uint64_t FieldVal;
  unsigned StartBit;

  // 0: Init
  // 1: Water (the bit value does not affect decoding)
  // 2: Island (well-known bit value needed for decoding)
  unsigned State = 0;

  unsigned FilterWidth = FilterBits.getBitWidth();
  for (unsigned i = 0; i != FilterWidth; ++i) {
    bool IsKnown = EncodingBits.Zero[i] || EncodingBits.One[i];
    bool Filtered = FilterBits.Zero[i] || FilterBits.One[i];
    switch (State) {
    default:
      llvm_unreachable("Unreachable code!");
    case 0:
    case 1:
      if (Filtered || !IsKnown) {
        State = 1; // Still in Water
      } else {
        State = 2; // Into the Island
        StartBit = i;
        FieldVal = static_cast<uint64_t>(EncodingBits.One[i]);
      }
      break;
    case 2:
      if (Filtered || !IsKnown) {
        State = 1; // Into the Water
        Islands.push_back({StartBit, i - StartBit, FieldVal});
      } else {
        State = 2; // Still in Island
        FieldVal |= static_cast<uint64_t>(EncodingBits.One[i])
                    << (i - StartBit);
      }
      break;
    }
  }
  // If we are still in Island after the loop, do some housekeeping.
  if (State == 2)
    Islands.push_back({StartBit, FilterWidth - StartBit, FieldVal});

  return Islands;
}

void DecoderTreeBuilder::emitBinaryParser(raw_ostream &OS, indent Indent,
                                          const OperandInfo &OpInfo) {
  // Special case for 'bits<0>'.
  if (OpInfo.Fields.empty() && !OpInfo.InitValue) {
    if (IgnoreNonDecodableOperands)
      return;
    assert(!OpInfo.Decoder.empty());
    // The operand has no encoding, so the corresponding argument is omitted.
    // This avoids confusion and allows the function to be overloaded if the
    // operand does have an encoding in other instructions.
    OS << Indent << "if (!Check(S, " << OpInfo.Decoder << "(MI, Decoder)))\n"
       << Indent << "  return MCDisassembler::Fail;\n";
    return;
  }

  if (OpInfo.Fields.empty() && OpInfo.InitValue && IgnoreFullyDefinedOperands)
    return;

  // We need to construct the encoding of the operand from pieces if it is not
  // encoded sequentially or has a non-zero constant part in the encoding.
  bool UseInsertBits = OpInfo.numFields() > 1 || OpInfo.InitValue.value_or(0);

  if (UseInsertBits) {
    OS << Indent << "tmp = 0x";
    OS.write_hex(OpInfo.InitValue.value_or(0));
    OS << ";\n";
  }

  for (const auto &[Base, Width, Offset] : OpInfo.fields()) {
    OS << Indent;
    if (UseInsertBits)
      OS << "insertBits(tmp, ";
    else
      OS << "tmp = ";
    OS << "fieldFromInstruction(insn, " << Base << ", " << Width << ')';
    if (UseInsertBits)
      OS << ", " << Offset << ", " << Width << ')';
    else if (Offset != 0)
      OS << " << " << Offset;
    OS << ";\n";
  }

  StringRef Decoder = OpInfo.Decoder;
  if (!Decoder.empty()) {
    OS << Indent << "if (!Check(S, " << Decoder
       << "(MI, tmp, Address, Decoder))) { "
       << (OpInfo.HasCompleteDecoder ? "" : "DecodeComplete = false; ")
       << "return MCDisassembler::Fail; }\n";
  } else {
    OS << Indent << "MI.addOperand(MCOperand::createImm(tmp));\n";
  }
}

void DecoderTreeBuilder::emitDecoder(raw_ostream &OS, indent Indent,
                                     const InstructionEncoding &Encoding) {
  // If a custom instruction decoder was specified, use that.
  StringRef DecoderMethod = Encoding.getDecoderMethod();
  if (!DecoderMethod.empty()) {
    OS << Indent << "if (!Check(S, " << DecoderMethod
       << "(MI, insn, Address, Decoder))) { "
       << (Encoding.hasCompleteDecoder() ? "" : "DecodeComplete = false; ")
       << "return MCDisassembler::Fail; }\n";
    return;
  }

  for (const OperandInfo &Op : Encoding.getOperands())
    emitBinaryParser(OS, Indent, Op);
}

unsigned
DecoderTreeBuilder::getDecoderIndex(const InstructionEncoding &Encoding) {
  // Build up the predicate string.
  SmallString<256> Decoder;
  // FIXME: emitDecoder() function can take a buffer directly rather than
  // a stream.
  raw_svector_ostream S(Decoder);
  indent Indent(UseFnTableInDecodeToMCInst ? 2 : 4);
  emitDecoder(S, Indent, Encoding);

  // Using the full decoder string as the key value here is a bit
  // heavyweight, but is effective. If the string comparisons become a
  // performance concern, we can implement a mangling of the predicate
  // data easily enough with a map back to the actual string. That's
  // overkill for now, though.

  // Make sure the predicate is in the table.
  DecoderSet &Decoders = TableInfo.Decoders;
  Decoders.insert(CachedHashString(Decoder));
  // Now figure out the index for when we write out the table.
  DecoderSet::const_iterator P = find(Decoders, Decoder.str());
  return std::distance(Decoders.begin(), P);
}

// If ParenIfBinOp is true, print a surrounding () if Val uses && or ||.
bool DecoderTreeBuilder::emitPredicateMatchAux(StringRef PredicateNamespace,
                                               const Init &Val,
                                               bool ParenIfBinOp,
                                               raw_ostream &OS) {
  if (const auto *D = dyn_cast<DefInit>(&Val)) {
    if (!D->getDef()->isSubClassOf("SubtargetFeature"))
      return true;
    OS << "Bits[" << PredicateNamespace << "::" << D->getAsString() << "]";
    return false;
  }
  if (const auto *D = dyn_cast<DagInit>(&Val)) {
    std::string Op = D->getOperator()->getAsString();
    if (Op == "not" && D->getNumArgs() == 1) {
      OS << '!';
      return emitPredicateMatchAux(PredicateNamespace, *D->getArg(0), true, OS);
    }
    if ((Op == "any_of" || Op == "all_of") && D->getNumArgs() > 0) {
      bool Paren = D->getNumArgs() > 1 && std::exchange(ParenIfBinOp, true);
      if (Paren)
        OS << '(';
      ListSeparator LS(Op == "any_of" ? " || " : " && ");
      for (auto *Arg : D->getArgs()) {
        OS << LS;
        if (emitPredicateMatchAux(PredicateNamespace, *Arg, ParenIfBinOp, OS))
          return true;
      }
      if (Paren)
        OS << ')';
      return false;
    }
  }
  return true;
}

bool DecoderTreeBuilder::emitPredicateMatch(
    StringRef PredicateNamespace, raw_ostream &OS,
    const InstructionEncoding &Encoding) {
  const ListInit *Predicates =
      Encoding.getRecord()->getValueAsListInit("Predicates");
  bool IsFirstEmission = true;
  for (unsigned i = 0; i < Predicates->size(); ++i) {
    const Record *Pred = Predicates->getElementAsRecord(i);
    if (!Pred->getValue("AssemblerMatcherPredicate"))
      continue;

    if (!isa<DagInit>(Pred->getValue("AssemblerCondDag")->getValue()))
      continue;

    if (!IsFirstEmission)
      OS << " && ";
    if (emitPredicateMatchAux(PredicateNamespace,
                              *Pred->getValueAsDag("AssemblerCondDag"),
                              Predicates->size() > 1, OS))
      PrintFatalError(Pred->getLoc(), "Invalid AssemblerCondDag!");
    IsFirstEmission = false;
  }
  return !Predicates->empty();
}

static bool doesOpcodeNeedPredicate(const InstructionEncoding &Encoding) {
  const ListInit *Predicates =
      Encoding.getRecord()->getValueAsListInit("Predicates");
  for (unsigned i = 0; i < Predicates->size(); ++i) {
    const Record *Pred = Predicates->getElementAsRecord(i);
    if (!Pred->getValue("AssemblerMatcherPredicate"))
      continue;

    if (isa<DagInit>(Pred->getValue("AssemblerCondDag")->getValue()))
      return true;
  }
  return false;
}

unsigned DecoderTreeBuilder::getPredicateIndex(
    const InstructionEncoding &Encoding) const {
  // Build up the predicate string.
  SmallString<256> Predicate;
  // FIXME: emitPredicateMatch() functions can take a buffer directly rather
  // than a stream.
  raw_svector_ostream PS(Predicate);
  emitPredicateMatch(Target.getName(), PS, Encoding);

  // Using the full predicate string as the key value here is a bit
  // heavyweight, but is effective. If the string comparisons become a
  // performance concern, we can implement a mangling of the predicate
  // data easily enough with a map back to the actual string. That's
  // overkill for now, though.

  // Make sure the predicate is in the table.
  PredicateSet &Predicates = TableInfo.Predicates;
  Predicates.insert(CachedHashString(Predicate));
  // Now figure out the index for when we write out the table.
  PredicateSet::const_iterator P = find(Predicates, Predicate);
  return std::distance(Predicates.begin(), P);
}

// reportRegion is a helper function for filterProcessor to mark a region as
// eligible for use as a filter region.
void DecoderTreeBuilder::reportRegion(
    ArrayRef<unsigned> EncodingIDs,
    std::vector<std::unique_ptr<Filter>> &Filters, bitAttr_t RA,
    unsigned StartBit, unsigned BitIndex, bool AllowMixed) const {
  if (AllowMixed ? RA == ATTR_MIXED : RA == ATTR_ALL_SET)
    Filters.push_back(std::make_unique<Filter>(Encodings, EncodingIDs, StartBit,
                                               BitIndex - StartBit));
}

std::unique_ptr<Filter> DecoderTreeBuilder::findBestFilter(
    ArrayRef<unsigned> EncodingIDs, const KnownBits &FilterBits,
    ArrayRef<bitAttr_t> BitAttrs, bool AllowMixed, bool Greedy) const {
  assert(EncodingIDs.size() >= 2 && "Nothing to filter");

  // Heuristics.  See also doFilter()'s "Heuristics" comment when num of
  // instructions is 3.
  if (AllowMixed && !Greedy) {
    assert(EncodingIDs.size() == 3);

    for (unsigned EncodingID : EncodingIDs) {
      const InstructionEncoding &Encoding = Encodings[EncodingID];
      KnownBits EncodingBits = Encoding.getMandatoryBits();

      // Look for islands of undecoded bits of any instruction.
      std::vector<Island> Islands = getIslands(EncodingBits, FilterBits);
      if (!Islands.empty()) {
        // Found an instruction with island(s).  Now just assign a filter.
        return std::make_unique<Filter>(
            Encodings, EncodingIDs, Islands[0].StartBit, Islands[0].NumBits);
      }
    }
  }

  // The regionAttr automaton consumes the bitAttrs automatons' state,
  // lowest-to-highest.
  //
  //   Input symbols: F(iltered), (all_)S(et), (all_)U(nset), M(ixed)
  //   States:        NONE, ALL_SET, MIXED
  //   Initial state: NONE
  //
  // (NONE) ----- F --> (NONE)
  // (NONE) ----- S --> (ALL_SET)     ; and set region start
  // (NONE) ----- U --> (NONE)
  // (NONE) ----- M --> (MIXED)       ; and set region start
  // (ALL_SET) -- F --> (NONE)        ; and report an ALL_SET region
  // (ALL_SET) -- S --> (ALL_SET)
  // (ALL_SET) -- U --> (NONE)        ; and report an ALL_SET region
  // (ALL_SET) -- M --> (MIXED)       ; and report an ALL_SET region
  // (MIXED) ---- F --> (NONE)        ; and report a MIXED region
  // (MIXED) ---- S --> (ALL_SET)     ; and report a MIXED region
  // (MIXED) ---- U --> (NONE)        ; and report a MIXED region
  // (MIXED) ---- M --> (MIXED)

  bitAttr_t RA = ATTR_NONE;
  unsigned StartBit = 0;

  std::vector<std::unique_ptr<Filter>> Filters;
  unsigned FilterWidth = FilterBits.getBitWidth();
  for (unsigned BitIndex = 0; BitIndex != FilterWidth; ++BitIndex) {
    bitAttr_t bitAttr = BitAttrs[BitIndex];

    assert(bitAttr != ATTR_NONE && "Bit without attributes");

    switch (RA) {
    case ATTR_NONE:
      switch (bitAttr) {
      case ATTR_FILTERED:
        break;
      case ATTR_ALL_SET:
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
        break;
      case ATTR_MIXED:
        StartBit = BitIndex;
        RA = ATTR_MIXED;
        break;
      default:
        llvm_unreachable("Unexpected bitAttr!");
      }
      break;
    case ATTR_ALL_SET:
      switch (bitAttr) {
      case ATTR_FILTERED:
        reportRegion(EncodingIDs, Filters, RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        break;
      case ATTR_ALL_UNSET:
        reportRegion(EncodingIDs, Filters, RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        reportRegion(EncodingIDs, Filters, RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_MIXED;
        break;
      default:
        llvm_unreachable("Unexpected bitAttr!");
      }
      break;
    case ATTR_MIXED:
      switch (bitAttr) {
      case ATTR_FILTERED:
        reportRegion(EncodingIDs, Filters, RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        reportRegion(EncodingIDs, Filters, RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
        reportRegion(EncodingIDs, Filters, RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        break;
      default:
        llvm_unreachable("Unexpected bitAttr!");
      }
      break;
    case ATTR_ALL_UNSET:
      llvm_unreachable("regionAttr state machine has no ATTR_UNSET state");
    case ATTR_FILTERED:
      llvm_unreachable("regionAttr state machine has no ATTR_FILTERED state");
    }
  }

  // At the end, if we're still in ALL_SET or MIXED states, report a region
  switch (RA) {
  case ATTR_NONE:
    break;
  case ATTR_FILTERED:
    break;
  case ATTR_ALL_SET:
    reportRegion(EncodingIDs, Filters, RA, StartBit, FilterWidth, AllowMixed);
    break;
  case ATTR_ALL_UNSET:
    break;
  case ATTR_MIXED:
    reportRegion(EncodingIDs, Filters, RA, StartBit, FilterWidth, AllowMixed);
    break;
  }

  // We have finished with the filter processings.  Now it's time to choose
  // the best performing filter.
  auto MaxIt = llvm::max_element(Filters, [](const std::unique_ptr<Filter> &A,
                                             const std::unique_ptr<Filter> &B) {
    return A->usefulness() < B->usefulness();
  });
  if (MaxIt == Filters.end() || (*MaxIt)->usefulness() == 0)
    return nullptr;
  return std::move(*MaxIt);
}

std::unique_ptr<Filter>
DecoderTreeBuilder::findBestFilter(ArrayRef<unsigned> EncodingIDs,
                                   const KnownBits &FilterBits) const {
  // We maintain BIT_WIDTH copies of the bitAttrs automaton.
  // The automaton consumes the corresponding bit from each
  // instruction.
  //
  //   Input symbols: 0, 1, _ (unset), and . (any of the above).
  //   States:        NONE, FILTERED, ALL_SET, ALL_UNSET, and MIXED.
  //   Initial state: NONE.
  //
  // (NONE) ------- [01] -> (ALL_SET)
  // (NONE) ------- _ ----> (ALL_UNSET)
  // (ALL_SET) ---- [01] -> (ALL_SET)
  // (ALL_SET) ---- _ ----> (MIXED)
  // (ALL_UNSET) -- [01] -> (MIXED)
  // (ALL_UNSET) -- _ ----> (ALL_UNSET)
  // (MIXED) ------ . ----> (MIXED)
  // (FILTERED)---- . ----> (FILTERED)

  unsigned FilterWidth = FilterBits.getBitWidth();
  SmallVector<bitAttr_t, 128> BitAttrs(FilterWidth, ATTR_NONE);

  // FILTERED bit positions provide no entropy and are not worthy of pursuing.
  // Filter::recurse() set either 1 or 0 for each position.
  for (unsigned BitIndex = 0; BitIndex != FilterWidth; ++BitIndex)
    if (FilterBits.Zero[BitIndex] || FilterBits.One[BitIndex])
      BitAttrs[BitIndex] = ATTR_FILTERED;

  for (unsigned EncodingID : EncodingIDs) {
    const InstructionEncoding &Encoding = Encodings[EncodingID];
    KnownBits EncodingBits = Encoding.getMandatoryBits();

    for (unsigned BitIndex = 0; BitIndex != FilterWidth; ++BitIndex) {
      bool IsKnown = EncodingBits.Zero[BitIndex] || EncodingBits.One[BitIndex];
      switch (BitAttrs[BitIndex]) {
      case ATTR_NONE:
        if (IsKnown)
          BitAttrs[BitIndex] = ATTR_ALL_SET;
        else
          BitAttrs[BitIndex] = ATTR_ALL_UNSET;
        break;
      case ATTR_ALL_SET:
        if (!IsKnown)
          BitAttrs[BitIndex] = ATTR_MIXED;
        break;
      case ATTR_ALL_UNSET:
        if (IsKnown)
          BitAttrs[BitIndex] = ATTR_MIXED;
        break;
      case ATTR_MIXED:
      case ATTR_FILTERED:
        break;
      }
    }
  }

  // Try regions of consecutive known bit values first.
  if (std::unique_ptr<Filter> F = findBestFilter(
          EncodingIDs, FilterBits, BitAttrs, /*AllowMixed=*/false))
    return F;

  // Then regions of mixed bits (both known and unitialized bit values allowed).
  if (std::unique_ptr<Filter> F = findBestFilter(EncodingIDs, FilterBits,
                                                 BitAttrs, /*AllowMixed=*/true))
    return F;

  // Heuristics to cope with conflict set {t2CMPrs, t2SUBSrr, t2SUBSrs} where
  // no single instruction for the maximum ATTR_MIXED region Inst{14-4} has a
  // well-known encoding pattern.  In such case, we backtrack and scan for the
  // the very first consecutive ATTR_ALL_SET region and assign a filter to it.
  if (EncodingIDs.size() == 3) {
    if (std::unique_ptr<Filter> F =
            findBestFilter(EncodingIDs, FilterBits, BitAttrs,
                           /*AllowMixed=*/true, /*Greedy=*/false))
      return F;
  }

  // There is a conflict we could not resolve.
  return nullptr;
}

void DecoderTreeBuilder::dump(ArrayRef<unsigned> EncodingIDs,
                              const KnownBits &FilterBits) const {
  indent Indent(4);
  // Helps to keep the output right-justified.
  unsigned PadToWidth = Encodings[EncodingIDs.back()].getBitWidth();

  // Dump filter stack.
  dumpStack(errs(), Indent, PadToWidth, FilterBits);

  // Dump encodings.
  for (unsigned EncodingID : EncodingIDs) {
    const InstructionEncoding &Encoding = Encodings[EncodingID];
    errs() << Indent << indent(PadToWidth - Encoding.getBitWidth());
    printKnownBits(errs(), Encoding.getMandatoryBits(), '_');
    errs() << "  " << Encoding.getName() << '\n';
  }
}

unsigned DecoderTableEmitter::computeNodeSize(const DecoderTreeNode *N) const {
  switch (N->getKind()) {
  case DecoderTreeNode::AnyOf: {
    const auto *AnyOf = static_cast<const AnyOfNode *>(N);
    unsigned Size = 0;
    for (const DecoderTreeNode *Child : drop_end(AnyOf->children())) {
      unsigned ChildSize = computeNodeSize(Child);
      Size += 1 + getULEB128Size(ChildSize) + ChildSize;
    }
    return Size + computeNodeSize(*std::prev(AnyOf->child_end()));
  }
  case DecoderTreeNode::AllOf: {
    const auto *AllOf = static_cast<const AllOfNode *>(N);
    unsigned Size = 0;
    for (const DecoderTreeNode *Child : AllOf->children())
      Size += computeNodeSize(Child);
    return Size;
  }
  case DecoderTreeNode::CheckField: {
    const auto *CheckField = static_cast<const CheckFieldNode *>(N);
    return 1 + getULEB128Size(CheckField->getStartBit()) + 1 +
           getULEB128Size(CheckField->getValue());
  }
  case DecoderTreeNode::SwitchField: {
    const auto *SwitchN = static_cast<const SwitchFieldNode *>(N);
    unsigned Size = 1 + getULEB128Size(SwitchN->getStartBit()) + 1;

    for (auto [Val, Child] : drop_end(SwitchN->cases())) {
      unsigned ChildSize = computeNodeSize(Child);
      Size += getULEB128Size(Val) + getULEB128Size(ChildSize) + ChildSize;
    }

    auto [Val, Child] = *std::prev(SwitchN->case_end());
    unsigned ChildSize = computeNodeSize(Child);
    Size += getULEB128Size(Val) + getULEB128Size(0) + ChildSize;
    return Size;
  }
  case DecoderTreeNode::CheckPredicate: {
    const auto *CheckPredicate = static_cast<const CheckPredicateNode *>(N);
    return 1 + getULEB128Size(CheckPredicate->getPredicateIndex());
  }
  case DecoderTreeNode::SoftFail: {
    const auto *SoftFail = static_cast<const SoftFailNode *>(N);
    return 1 + getULEB128Size(SoftFail->getPositiveMask()) +
           getULEB128Size(SoftFail->getNegativeMask());
  }
  case DecoderTreeNode::Decode: {
    const auto *Decode = static_cast<const DecodeNode *>(N);
    const InstructionEncoding &Encoding = Encodings[Decode->getEncodingID()];
    const Record *InstDef = Encoding.getInstruction()->TheDef;
    unsigned InstOpcode = Target.getInstrIntValue(InstDef);
    return 1 + getULEB128Size(InstOpcode) +
           getULEB128Size(Decode->getDecoderIndex());
  }
  }
  llvm_unreachable("Unknown node kind");
}

void DecoderTableEmitter::emitAnyOfNode(const AnyOfNode *N, indent Indent) {
  for (const DecoderTreeNode *Child : drop_end(N->children())) {
    emitOpcode("OPC_Scope");
    emitULEB128(computeNodeSize(Child));

    emitComment(Indent) << "{\n";
    emitNode(Child, Indent + 1);
    emitComment(Indent) << "}\n";
  }

  const DecoderTreeNode *Child = *std::prev(N->child_end());
  emitNode(Child, Indent);
}

void DecoderTableEmitter::emitAllOfNode(const AllOfNode *N, indent Indent) {
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
  unsigned PredicateIndex = N->getPredicateIndex();

  emitOpcode("OPC_CheckPredicate");
  emitULEB128(PredicateIndex);

  emitComment(Indent) << "check predicate " << PredicateIndex << "\n";
}

void DecoderTableEmitter::emitSoftFailNode(const SoftFailNode *N,
                                           indent Indent) {
  uint64_t PositiveMask = N->getPositiveMask();
  uint64_t NegativeMask = N->getNegativeMask();

  emitOpcode("OPC_SoftFail");
  emitULEB128(PositiveMask);
  emitULEB128(NegativeMask);

  emitComment(Indent) << "check softfail";
  OS << " pos=" << format_hex(PositiveMask, 10);
  OS << " neg=" << format_hex(NegativeMask, 10) << '\n';
}

void DecoderTableEmitter::emitDecodeNode(const DecodeNode *N, indent Indent) {
  const InstructionEncoding &Encoding = Encodings[N->getEncodingID()];
  const Record *InstDef = Encoding.getInstruction()->TheDef;
  unsigned InstOpcode = Target.getInstrIntValue(InstDef);
  unsigned DecoderIndex = N->getDecoderIndex();

  emitOpcode(Encoding.hasCompleteDecoder() ? "OPC_Decode" : "OPC_TryDecode");
  emitULEB128(InstOpcode);
  emitULEB128(DecoderIndex);

  emitComment(Indent);
  if (!Encoding.hasCompleteDecoder())
    OS << "try ";
  OS << "decode to " << Encoding.getName() << " using decoder " << DecoderIndex
     << '\n';
}

void DecoderTableEmitter::emitNode(const DecoderTreeNode *N, indent Indent) {
  switch (N->getKind()) {
  case DecoderTreeNode::AnyOf:
    return emitAnyOfNode(static_cast<const AnyOfNode *>(N), Indent);
  case DecoderTreeNode::AllOf:
    return emitAllOfNode(static_cast<const AllOfNode *>(N), Indent);
  case DecoderTreeNode::SwitchField:
    return emitSwitchFieldNode(static_cast<const SwitchFieldNode *>(N), Indent);
  case DecoderTreeNode::CheckField:
    return emitCheckFieldNode(static_cast<const CheckFieldNode *>(N), Indent);
  case DecoderTreeNode::CheckPredicate:
    HasCheckPredicate = true;
    return emitCheckPredicateNode(static_cast<const CheckPredicateNode *>(N),
                                  Indent);
  case DecoderTreeNode::SoftFail:
    HasSoftFail = true;
    return emitSoftFailNode(static_cast<const SoftFailNode *>(N), Indent);
  case DecoderTreeNode::Decode:
    HasTryDecode |=
        Encodings[static_cast<const DecodeNode *>(N)->getEncodingID()]
            .hasCompleteDecoder();
    return emitDecodeNode(static_cast<const DecodeNode *>(N), Indent);
  }
  llvm_unreachable("Unknown node kind");
}

void DecoderTableEmitter::emitTable(StringRef TableName, unsigned BitWidth,
                                    const DecoderTreeNode *Root) {
  unsigned TableSize = computeTableSize(BitWidth, Root);
  OS << "static const uint8_t " << TableName << "[" << TableSize << "] = {\n";

  // Calculate the number of decimal places for table indices.
  // This is simply log10 of the table size.
  IndexWidth = 1;
  for (unsigned S = TableSize; S /= 10;)
    ++IndexWidth;

  CurrentIndex = 0;

  // When specializing decoders per bit width, each decoder table will begin
  // with the bitwidth for that table.
  if (SpecializeDecodersPerBitwidth) {
    emitStartLine();
    emitULEB128(BitWidth);
    emitComment(indent(0)) << "BitWidth " << BitWidth << '\n';
  }

  emitNode(Root, indent(0));
  assert(CurrentIndex == TableSize &&
         "The size of the emitted table differs from the calculated one");

  OS << "};\n";
}

static std::string findOperandDecoderMethod(const Record *Record) {
  std::string Decoder;

  const RecordVal *DecoderString = Record->getValue("DecoderMethod");
  const StringInit *String =
      DecoderString ? dyn_cast<StringInit>(DecoderString->getValue()) : nullptr;
  if (String) {
    Decoder = String->getValue().str();
    if (!Decoder.empty())
      return Decoder;
  }

  if (Record->isSubClassOf("RegisterOperand"))
    // Allows use of a DecoderMethod in referenced RegisterClass if set.
    return findOperandDecoderMethod(Record->getValueAsDef("RegClass"));

  if (Record->isSubClassOf("RegisterClass")) {
    Decoder = "Decode" + Record->getName().str() + "RegisterClass";
  } else if (Record->isSubClassOf("PointerLikeRegClass")) {
    Decoder = "DecodePointerLikeRegClass" +
              utostr(Record->getValueAsInt("RegClassKind"));
  }

  return Decoder;
}

OperandInfo getOpInfo(const Record *TypeRecord) {
  const RecordVal *HasCompleteDecoderVal =
      TypeRecord->getValue("hasCompleteDecoder");
  const BitInit *HasCompleteDecoderBit =
      HasCompleteDecoderVal
          ? dyn_cast<BitInit>(HasCompleteDecoderVal->getValue())
          : nullptr;
  bool HasCompleteDecoder =
      HasCompleteDecoderBit ? HasCompleteDecoderBit->getValue() : true;

  return OperandInfo(findOperandDecoderMethod(TypeRecord), HasCompleteDecoder);
}

void InstructionEncoding::parseVarLenEncoding(const VarLenInst &VLI) {
  InstBits = KnownBits(VLI.size());
  SoftFailMask = APInt(VLI.size(), 0);

  // Parse Inst field.
  unsigned I = 0;
  for (const EncodingSegment &S : VLI) {
    if (const auto *SegmentBits = dyn_cast<BitsInit>(S.Value)) {
      for (const Init *V : SegmentBits->getBits()) {
        if (const auto *B = dyn_cast<BitInit>(V)) {
          if (B->getValue())
            InstBits.One.setBit(I);
          else
            InstBits.Zero.setBit(I);
        }
        ++I;
      }
    } else if (const auto *B = dyn_cast<BitInit>(S.Value)) {
      if (B->getValue())
        InstBits.One.setBit(I);
      else
        InstBits.Zero.setBit(I);
      ++I;
    } else {
      I += S.BitWidth;
    }
  }
  assert(I == VLI.size());
}

void InstructionEncoding::parseFixedLenEncoding(
    const BitsInit &RecordInstBits) {
  // For fixed length instructions, sometimes the `Inst` field specifies more
  // bits than the actual size of the instruction, which is specified in `Size`.
  // In such cases, we do some basic validation and drop the upper bits.
  unsigned BitWidth = EncodingDef->getValueAsInt("Size") * 8;
  unsigned InstNumBits = RecordInstBits.getNumBits();

  // Returns true if all bits in `Bits` are zero or unset.
  auto CheckAllZeroOrUnset = [&](ArrayRef<const Init *> Bits,
                                 const RecordVal *Field) {
    bool AllZeroOrUnset = llvm::all_of(Bits, [](const Init *Bit) {
      if (const auto *BI = dyn_cast<BitInit>(Bit))
        return !BI->getValue();
      return isa<UnsetInit>(Bit);
    });
    if (AllZeroOrUnset)
      return;
    PrintNote([Field](raw_ostream &OS) { Field->print(OS); });
    PrintFatalError(EncodingDef, Twine(Name) + ": Size is " + Twine(BitWidth) +
                                     " bits, but " + Field->getName() +
                                     " bits beyond that are    not zero/unset");
  };

  if (InstNumBits < BitWidth)
    PrintFatalError(EncodingDef, Twine(Name) + ": Size is " + Twine(BitWidth) +
                                     " bits, but Inst specifies only " +
                                     Twine(InstNumBits) + " bits");

  if (InstNumBits > BitWidth) {
    // Ensure that all the bits beyond 'Size' are 0 or unset (i.e., carry no
    // actual encoding).
    ArrayRef<const Init *> UpperBits =
        RecordInstBits.getBits().drop_front(BitWidth);
    const RecordVal *InstField = EncodingDef->getValue("Inst");
    CheckAllZeroOrUnset(UpperBits, InstField);
  }

  ArrayRef<const Init *> ActiveInstBits =
      RecordInstBits.getBits().take_front(BitWidth);
  InstBits = KnownBits(BitWidth);
  SoftFailMask = APInt(BitWidth, 0);

  // Parse Inst field.
  for (auto [I, V] : enumerate(ActiveInstBits)) {
    if (const auto *B = dyn_cast<BitInit>(V)) {
      if (B->getValue())
        InstBits.One.setBit(I);
      else
        InstBits.Zero.setBit(I);
    }
  }

  // Parse SoftFail field.
  const RecordVal *SoftFailField = EncodingDef->getValue("SoftFail");
  if (!SoftFailField)
    return;

  const auto *SFBits = dyn_cast<BitsInit>(SoftFailField->getValue());
  if (!SFBits || SFBits->getNumBits() != InstNumBits) {
    PrintNote(EncodingDef->getLoc(), "in record");
    PrintFatalError(SoftFailField,
                    formatv("SoftFail field, if defined, must be "
                            "of the same type as Inst, which is bits<{}>",
                            InstNumBits));
  }

  if (InstNumBits > BitWidth) {
    // Ensure that all upper bits of `SoftFail` are 0 or unset.
    ArrayRef<const Init *> UpperBits = SFBits->getBits().drop_front(BitWidth);
    CheckAllZeroOrUnset(UpperBits, SoftFailField);
  }

  ArrayRef<const Init *> ActiveSFBits = SFBits->getBits().take_front(BitWidth);
  for (auto [I, V] : enumerate(ActiveSFBits)) {
    if (const auto *B = dyn_cast<BitInit>(V); B && B->getValue()) {
      if (!InstBits.Zero[I] && !InstBits.One[I]) {
        PrintNote(EncodingDef->getLoc(), "in record");
        PrintError(SoftFailField,
                   formatv("SoftFail{{{0}} = 1 requires Inst{{{0}} "
                           "to be fully defined (0 or 1, not '?')",
                           I));
      }
      SoftFailMask.setBit(I);
    }
  }
}

void InstructionEncoding::parseVarLenOperands(const VarLenInst &VLI) {
  SmallVector<int> TiedTo;

  for (const auto &[Idx, Op] : enumerate(Inst->Operands)) {
    if (Op.MIOperandInfo && Op.MIOperandInfo->getNumArgs() > 0)
      for (auto *Arg : Op.MIOperandInfo->getArgs())
        Operands.push_back(getOpInfo(cast<DefInit>(Arg)->getDef()));
    else
      Operands.push_back(getOpInfo(Op.Rec));

    int TiedReg = Op.getTiedRegister();
    TiedTo.push_back(-1);
    if (TiedReg != -1) {
      TiedTo[Idx] = TiedReg;
      TiedTo[TiedReg] = Idx;
    }
  }

  unsigned CurrBitPos = 0;
  for (const auto &EncodingSegment : VLI) {
    unsigned Offset = 0;
    StringRef OpName;

    if (const StringInit *SI = dyn_cast<StringInit>(EncodingSegment.Value)) {
      OpName = SI->getValue();
    } else if (const DagInit *DI = dyn_cast<DagInit>(EncodingSegment.Value)) {
      OpName = cast<StringInit>(DI->getArg(0))->getValue();
      Offset = cast<IntInit>(DI->getArg(2))->getValue();
    }

    if (!OpName.empty()) {
      auto OpSubOpPair = Inst->Operands.parseOperandName(OpName);
      unsigned OpIdx = Inst->Operands.getFlattenedOperandNumber(OpSubOpPair);
      Operands[OpIdx].addField(CurrBitPos, EncodingSegment.BitWidth, Offset);
      if (!EncodingSegment.CustomDecoder.empty())
        Operands[OpIdx].Decoder = EncodingSegment.CustomDecoder.str();

      int TiedReg = TiedTo[OpSubOpPair.first];
      if (TiedReg != -1) {
        unsigned OpIdx = Inst->Operands.getFlattenedOperandNumber(
            {TiedReg, OpSubOpPair.second});
        Operands[OpIdx].addField(CurrBitPos, EncodingSegment.BitWidth, Offset);
      }
    }

    CurrBitPos += EncodingSegment.BitWidth;
  }
}

static void debugDumpRecord(const Record &Rec) {
  // Dump the record, so we can see what's going on.
  PrintNote([&Rec](raw_ostream &OS) {
    OS << "Dumping record for previous error:\n";
    OS << Rec;
  });
}

/// For an operand field named OpName: populate OpInfo.InitValue with the
/// constant-valued bit values, and OpInfo.Fields with the ranges of bits to
/// insert from the decoded instruction.
static void addOneOperandFields(const Record *EncodingDef,
                                const BitsInit &InstBits,
                                std::map<StringRef, StringRef> &TiedNames,
                                const Record *OpRec, StringRef OpName,
                                OperandInfo &OpInfo) {
  // Find a field with the operand's name.
  const RecordVal *OpEncodingField = EncodingDef->getValue(OpName);

  // If there is no such field, try tied operand's name.
  if (!OpEncodingField) {
    if (auto I = TiedNames.find(OpName); I != TiedNames.end())
      OpEncodingField = EncodingDef->getValue(I->second);

    // If still no luck, the old behavior is to not decode this operand
    // automatically and let the target do it. This is error-prone, so
    // the new behavior is to report an error.
    if (!OpEncodingField) {
      if (!IgnoreNonDecodableOperands)
        PrintError(EncodingDef->getLoc(),
                   "could not find field for operand '" + OpName + "'");
      return;
    }
  }

  // Some or all bits of the operand may be required to be 0 or 1 depending
  // on the instruction's encoding. Collect those bits.
  if (const auto *OpBit = dyn_cast<BitInit>(OpEncodingField->getValue())) {
    OpInfo.InitValue = OpBit->getValue();
    return;
  }
  if (const auto *OpBits = dyn_cast<BitsInit>(OpEncodingField->getValue())) {
    if (OpBits->getNumBits() == 0) {
      if (OpInfo.Decoder.empty()) {
        PrintError(EncodingDef->getLoc(), "operand '" + OpName + "' of type '" +
                                              OpRec->getName() +
                                              "' must have a decoder method");
      }
      return;
    }
    for (unsigned I = 0; I < OpBits->getNumBits(); ++I) {
      if (const auto *OpBit = dyn_cast<BitInit>(OpBits->getBit(I)))
        OpInfo.InitValue = OpInfo.InitValue.value_or(0) |
                           static_cast<uint64_t>(OpBit->getValue()) << I;
    }
  }

  // Find out where the variable bits of the operand are encoded. The bits don't
  // have to be consecutive or in ascending order. For example, an operand could
  // be encoded as follows:
  //
  //  7    6      5      4    3    2      1    0
  // {1, op{5}, op{2}, op{1}, 0, op{4}, op{3}, ?}
  //
  // In this example the operand is encoded in three segments:
  //
  //           Base Width Offset
  // op{2...1}   4    2     1
  // op{4...3}   1    2     3
  // op{5}       6    1     5
  //
  for (unsigned I = 0, J = 0; I != InstBits.getNumBits(); I = J) {
    const VarInit *Var;
    unsigned Offset = 0;
    for (; J != InstBits.getNumBits(); ++J) {
      const Init *BitJ = InstBits.getBit(J);
      if (const auto *VBI = dyn_cast<VarBitInit>(BitJ)) {
        Var = dyn_cast<VarInit>(VBI->getBitVar());
        if (I == J)
          Offset = VBI->getBitNum();
        else if (VBI->getBitNum() != Offset + J - I)
          break;
      } else {
        Var = dyn_cast<VarInit>(BitJ);
      }
      if (!Var ||
          (Var->getName() != OpName && Var->getName() != TiedNames[OpName]))
        break;
    }
    if (I == J)
      ++J;
    else
      OpInfo.addField(I, J - I, Offset);
  }
}

void InstructionEncoding::parseFixedLenOperands(const BitsInit &Bits) {
  // Search for tied operands, so that we can correctly instantiate
  // operands that are not explicitly represented in the encoding.
  std::map<StringRef, StringRef> TiedNames;
  for (const auto &Op : Inst->Operands) {
    for (const auto &[J, CI] : enumerate(Op.Constraints)) {
      if (!CI.isTied())
        continue;
      std::pair<unsigned, unsigned> SO =
          Inst->Operands.getSubOperandNumber(CI.getTiedOperand());
      StringRef TiedName = Inst->Operands[SO.first].SubOpNames[SO.second];
      if (TiedName.empty())
        TiedName = Inst->Operands[SO.first].Name;
      StringRef MyName = Op.SubOpNames[J];
      if (MyName.empty())
        MyName = Op.Name;

      TiedNames[MyName] = TiedName;
      TiedNames[TiedName] = MyName;
    }
  }

  // For each operand, see if we can figure out where it is encoded.
  for (const CGIOperandList::OperandInfo &Op : Inst->Operands) {
    // Lookup the decoder method and construct a new OperandInfo to hold our
    // result.
    OperandInfo OpInfo = getOpInfo(Op.Rec);

    // If we have named sub-operands...
    if (Op.MIOperandInfo && !Op.SubOpNames[0].empty()) {
      // Then there should not be a custom decoder specified on the top-level
      // type.
      if (!OpInfo.Decoder.empty()) {
        PrintError(EncodingDef,
                   "DecoderEmitter: operand \"" + Op.Name + "\" has type \"" +
                       Op.Rec->getName() +
                       "\" with a custom DecoderMethod, but also named "
                       "sub-operands.");
        continue;
      }

      // Decode each of the sub-ops separately.
      for (auto [SubOpName, SubOp] :
           zip_equal(Op.SubOpNames, Op.MIOperandInfo->getArgs())) {
        const Record *SubOpRec = cast<DefInit>(SubOp)->getDef();
        OperandInfo SubOpInfo = getOpInfo(SubOpRec);
        addOneOperandFields(EncodingDef, Bits, TiedNames, SubOpRec, SubOpName,
                            SubOpInfo);
        Operands.push_back(std::move(SubOpInfo));
      }
      continue;
    }

    // Otherwise, if we have an operand with sub-operands, but they aren't
    // named...
    if (Op.MIOperandInfo && OpInfo.Decoder.empty()) {
      // If we have sub-ops, we'd better have a custom decoder.
      // (Otherwise we don't know how to populate them properly...)
      if (Op.MIOperandInfo->getNumArgs()) {
        PrintError(EncodingDef,
                   "DecoderEmitter: operand \"" + Op.Name +
                       "\" has non-empty MIOperandInfo, but doesn't "
                       "have a custom decoder!");
        debugDumpRecord(*EncodingDef);
        continue;
      }
    }

    addOneOperandFields(EncodingDef, Bits, TiedNames, Op.Rec, Op.Name, OpInfo);
    Operands.push_back(std::move(OpInfo));
  }
}

InstructionEncoding::InstructionEncoding(const Record *EncodingDef,
                                         const CodeGenInstruction *Inst)
    : EncodingDef(EncodingDef), Inst(Inst) {
  const Record *InstDef = Inst->TheDef;

  // Give this encoding a name.
  if (EncodingDef != InstDef)
    Name = (EncodingDef->getName() + Twine(':')).str();
  Name.append(InstDef->getName());

  DecoderNamespace = EncodingDef->getValueAsString("DecoderNamespace");
  DecoderMethod = EncodingDef->getValueAsString("DecoderMethod");
  if (!DecoderMethod.empty())
    HasCompleteDecoder = EncodingDef->getValueAsBit("hasCompleteDecoder");

  const RecordVal *InstField = EncodingDef->getValue("Inst");
  if (const auto *DI = dyn_cast<DagInit>(InstField->getValue())) {
    VarLenInst VLI(DI, InstField);
    parseVarLenEncoding(VLI);
    // If the encoding has a custom decoder, don't bother parsing the operands.
    if (DecoderMethod.empty())
      parseVarLenOperands(VLI);
  } else {
    const auto *BI = cast<BitsInit>(InstField->getValue());
    parseFixedLenEncoding(*BI);
    // If the encoding has a custom decoder, don't bother parsing the operands.
    if (DecoderMethod.empty())
      parseFixedLenOperands(*BI);
  }

  if (DecoderMethod.empty()) {
    // A generated decoder is always successful if none of the operand
    // decoders can fail (all are always successful).
    HasCompleteDecoder = all_of(Operands, [](const OperandInfo &Op) {
      // By default, a generated operand decoder is assumed to always succeed.
      // This can be overridden by the user.
      return Op.Decoder.empty() || Op.HasCompleteDecoder;
    });
  }
}

// emitDecodeInstruction - Emit the templated helper function
// decodeInstruction().
static void emitDecodeInstruction(formatted_raw_ostream &OS, bool IsVarLenInst,
                                  bool HasCheckPredicate, bool HasSoftFail,
                                  bool HasTryDecode) {
  OS << R"(
template <typename InsnType>
static DecodeStatus decodeInstruction(const uint8_t DecodeTable[], MCInst &MI,
                                      InsnType insn, uint64_t Address,
                                      const MCDisassembler *DisAsm,
                                      const MCSubtargetInfo &STI)";
  if (IsVarLenInst) {
    OS << ",\n                                      "
          "llvm::function_ref<void(APInt &, uint64_t)> makeUp";
  }
  OS << ") {\n";
  if (HasCheckPredicate)
    OS << "  const FeatureBitset &Bits = STI.getFeatureBits();\n";
  OS << "  using namespace llvm::MCD;\n";
  OS << "  const uint8_t *Ptr = DecodeTable;\n";

  if (SpecializeDecodersPerBitwidth) {
    // Fail with a fatal error if decoder table's bitwidth does not match
    // `InsnType` bitwidth.
    OS << R"(
  [[maybe_unused]] uint32_t BitWidth = decodeULEB128AndIncUnsafe(Ptr);
  assert(InsnBitWidth<InsnType> == BitWidth &&
         "Table and instruction bitwidth mismatch");
)";
  }

  OS << R"(
  SmallVector<const uint8_t *, 8> ScopeStack;
  DecodeStatus S = MCDisassembler::Success;
  while (true) {
    ptrdiff_t Loc = Ptr - DecodeTable;
    const uint8_t DecoderOp = *Ptr++;
    switch (DecoderOp) {
    default:
      errs() << Loc << ": Unexpected decode table opcode: "
             << (int)DecoderOp << '\n';
      return MCDisassembler::Fail;
    case MCD::OPC_Scope: {
      unsigned NumToSkip = decodeULEB128AndIncUnsafe(Ptr);
      const uint8_t *SkipTo = Ptr + NumToSkip;
      ScopeStack.push_back(SkipTo);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Scope(" << SkipTo - DecodeTable
                        << ")\n");
      break;
    }
    case MCD::OPC_SwitchField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(Ptr);
      unsigned Len = *Ptr++;)";
  if (IsVarLenInst)
    OS << "\n      makeUp(insn, Start + Len);";
  OS << R"(
      uint64_t FieldVal = fieldFromInstruction(insn, Start, Len);
      uint64_t CaseVal;
      unsigned CaseSize;
      while (true) {
        CaseVal = decodeULEB128AndIncUnsafe(Ptr);
        CaseSize = decodeULEB128AndIncUnsafe(Ptr);
        if (FieldVal == CaseVal || !CaseSize)
          break;
        Ptr += CaseSize;
      }
      if (FieldVal == CaseVal) {
        LLVM_DEBUG(dbgs() << Loc << ": OPC_SwitchField(" << Start << ", " << Len
                          << "): " << FieldVal << '\n');
      } else {
        if (ScopeStack.empty()) {
          LLVM_DEBUG(dbgs() << "returning Fail\n");
          return MCDisassembler::Fail;
        }
        Ptr = ScopeStack.pop_back_val();
        LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
      }
      break;
    }
    case MCD::OPC_CheckField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(Ptr);
      unsigned Len = *Ptr;)";
  if (IsVarLenInst)
    OS << "\n      makeUp(insn, Start + Len);";
  OS << R"(
      uint64_t FieldValue = fieldFromInstruction(insn, Start, Len);
      // Decode the field value.
      unsigned PtrLen = 0;
      uint64_t ExpectedValue = decodeULEB128(++Ptr, &PtrLen);
      Ptr += PtrLen;
      bool Failed = ExpectedValue != FieldValue;

      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckField(" << Start << ", " << Len
                        << ", " << ExpectedValue << "): FieldValue = "
                        << FieldValue << ", ExpectedValue = " << ExpectedValue
                        << ": " << (Failed ? "FAIL, " : "PASS\n"););
      if (Failed) {
        if (ScopeStack.empty()) {
          LLVM_DEBUG(dbgs() << "returning Fail\n");
          return MCDisassembler::Fail;
        }
        Ptr = ScopeStack.pop_back_val();
        LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
      }
      break;
    })";
  if (HasCheckPredicate) {
    OS << R"(
    case MCD::OPC_CheckPredicate: {
      // Decode the Predicate Index value.
      unsigned PIdx = decodeULEB128AndIncUnsafe(Ptr);
      // Check the predicate.
      bool Failed = !checkDecoderPredicate(PIdx, Bits);

      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckPredicate(" << PIdx << "): "
                        << (Failed ? "FAIL, " : "PASS\n"););

      if (Failed) {
        if (ScopeStack.empty()) {
          LLVM_DEBUG(dbgs() << "returning Fail\n");
          return MCDisassembler::Fail;
        }
        Ptr = ScopeStack.pop_back_val();
        LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
      }
      break;
    })";
  }
  OS << R"(
    case MCD::OPC_Decode: {
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128AndIncUnsafe(Ptr);
      unsigned DecodeIdx = decodeULEB128AndIncUnsafe(Ptr);

      MI.clear();
      MI.setOpcode(Opc);
      bool DecodeComplete;)";
  if (IsVarLenInst) {
    OS << "\n      unsigned Len = InstrLenTable[Opc];\n"
       << "      makeUp(insn, Len);";
  }
  OS << R"(
      S = decodeToMCInst(DecodeIdx, S, insn, MI, Address, DisAsm, DecodeComplete);
      assert(DecodeComplete);

      LLVM_DEBUG(dbgs() << Loc << ": OPC_Decode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": "
                   << (S != MCDisassembler::Fail ? "PASS\n" : "FAIL\n"));
      return S;
    })";
  if (HasTryDecode) {
    OS << R"(
    case MCD::OPC_TryDecode: {
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128AndIncUnsafe(Ptr);
      unsigned DecodeIdx = decodeULEB128AndIncUnsafe(Ptr);

      // Perform the decode operation.
      MCInst TmpMI;
      TmpMI.setOpcode(Opc);
      bool DecodeComplete;
      S = decodeToMCInst(DecodeIdx, S, insn, TmpMI, Address, DisAsm, DecodeComplete);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_TryDecode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": ");

      if (DecodeComplete) {
        // Decoding complete.
        LLVM_DEBUG(dbgs() << (S != MCDisassembler::Fail ? "PASS\n" : "FAIL\n"));
        MI = TmpMI;
        return S;
      }
      assert(S == MCDisassembler::Fail);
      if (ScopeStack.empty()) {
        LLVM_DEBUG(dbgs() << "FAIL, returning FAIL\n");
        return MCDisassembler::Fail;
      }
      Ptr = ScopeStack.pop_back_val();
      LLVM_DEBUG(dbgs() << "FAIL, continuing at " << Ptr - DecodeTable << '\n');
      // Reset decode status. This also drops a SoftFail status that could be
      // set before the decode attempt.
      S = MCDisassembler::Success;
      break;
    })";
  }
  if (HasSoftFail) {
    OS << R"(
    case MCD::OPC_SoftFail: {
      // Decode the mask values.
      uint64_t PositiveMask = decodeULEB128AndIncUnsafe(Ptr);
      uint64_t NegativeMask = decodeULEB128AndIncUnsafe(Ptr);
      bool Failed = (insn & PositiveMask) != 0 || (~insn & NegativeMask) != 0;
      if (Failed)
        S = MCDisassembler::SoftFail;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_SoftFail: " << (Failed ? "FAIL\n" : "PASS\n"));
      break;
    })";
  }
  OS << R"(
    }
  }
  llvm_unreachable("bogosity detected in disassembler state machine!");
}

)";
}

/// Collects all HwModes referenced by the target for encoding purposes.
void DecoderEmitter::collectHwModesReferencedForEncodings(
    std::vector<unsigned> &HwModeIDs,
    NamespacesHwModesMap &NamespacesWithHwModes) const {
  SmallBitVector BV(CGH.getNumModeIds());
  for (const auto &MS : CGH.getHwModeSelects()) {
    for (auto [HwModeID, EncodingDef] : MS.second.Items) {
      if (EncodingDef->isSubClassOf("InstructionEncoding")) {
        StringRef DecoderNamespace =
            EncodingDef->getValueAsString("DecoderNamespace");
        NamespacesWithHwModes[DecoderNamespace].insert(HwModeID);
        BV.set(HwModeID);
      }
    }
  }
  // FIXME: Can't do `HwModeIDs.assign(BV.set_bits_begin(), BV.set_bits_end())`
  //   because const_set_bits_iterator_impl is not copy-assignable.
  //   This breaks some MacOS builds.
  llvm::copy(BV.set_bits(), std::back_inserter(HwModeIDs));
}

void DecoderEmitter::handleHwModesUnrelatedEncodings(
    unsigned EncodingID, ArrayRef<unsigned> HwModeIDs,
    NamespacesHwModesMap &NamespacesWithHwModes) {
  switch (DecoderEmitterSuppressDuplicates) {
  case SUPPRESSION_DISABLE: {
    for (unsigned HwModeID : HwModeIDs)
      EncodingIDsByHwMode[HwModeID].push_back(EncodingID);
    break;
  }
  case SUPPRESSION_LEVEL1: {
    StringRef DecoderNamespace = Encodings[EncodingID].getDecoderNamespace();
    auto It = NamespacesWithHwModes.find(DecoderNamespace);
    if (It != NamespacesWithHwModes.end()) {
      for (unsigned HwModeID : It->second)
        EncodingIDsByHwMode[HwModeID].push_back(EncodingID);
    } else {
      // Only emit the encoding once, as it's DecoderNamespace doesn't
      // contain any HwModes.
      EncodingIDsByHwMode[DefaultMode].push_back(EncodingID);
    }
    break;
  }
  case SUPPRESSION_LEVEL2:
    EncodingIDsByHwMode[DefaultMode].push_back(EncodingID);
    break;
  }
}

/// Checks if the given target-specific non-pseudo instruction
/// is a candidate for decoding.
static bool isDecodableInstruction(const Record *InstDef) {
  return !InstDef->getValueAsBit("isAsmParserOnly") &&
         !InstDef->getValueAsBit("isCodeGenOnly");
}

/// Checks if the given encoding is valid.
static bool isValidEncoding(const Record *EncodingDef) {
  const RecordVal *InstField = EncodingDef->getValue("Inst");
  if (!InstField)
    return false;

  if (const auto *InstInit = dyn_cast<BitsInit>(InstField->getValue())) {
    // Fixed-length encoding. Size must be non-zero.
    if (!EncodingDef->getValueAsInt("Size"))
      return false;

    // At least one of the encoding bits must be complete (not '?').
    // FIXME: This should take SoftFail field into account.
    return !InstInit->allInComplete();
  }

  if (const auto *InstInit = dyn_cast<DagInit>(InstField->getValue())) {
    // Variable-length encoding.
    // At least one of the encoding bits must be complete (not '?').
    VarLenInst VLI(InstInit, InstField);
    return !all_of(VLI, [](const EncodingSegment &Segment) {
      return isa<UnsetInit>(Segment.Value);
    });
  }

  // Inst field is neither BitsInit nor DagInit. This is something unsupported.
  return false;
}

/// Parses all InstructionEncoding instances and fills internal data structures.
void DecoderEmitter::parseInstructionEncodings() {
  // First, collect all encoding-related HwModes referenced by the target.
  // And establish a mapping table between DecoderNamespace and HwMode.
  // If HwModeNames is empty, add the default mode so we always have one HwMode.
  std::vector<unsigned> HwModeIDs;
  NamespacesHwModesMap NamespacesWithHwModes;
  collectHwModesReferencedForEncodings(HwModeIDs, NamespacesWithHwModes);
  if (HwModeIDs.empty())
    HwModeIDs.push_back(DefaultMode);

  ArrayRef<const CodeGenInstruction *> Instructions =
      Target.getTargetNonPseudoInstructions();
  Encodings.reserve(Instructions.size());

  for (const CodeGenInstruction *Inst : Instructions) {
    const Record *InstDef = Inst->TheDef;
    if (!isDecodableInstruction(InstDef)) {
      ++NumEncodingsLackingDisasm;
      continue;
    }

    if (const Record *RV = InstDef->getValueAsOptionalDef("EncodingInfos")) {
      EncodingInfoByHwMode EBM(RV, CGH);
      for (auto [HwModeID, EncodingDef] : EBM) {
        if (!isValidEncoding(EncodingDef)) {
          // TODO: Should probably give a warning.
          ++NumEncodingsOmitted;
          continue;
        }
        unsigned EncodingID = Encodings.size();
        Encodings.emplace_back(EncodingDef, Inst);
        EncodingIDsByHwMode[HwModeID].push_back(EncodingID);
      }
      continue; // Ignore encoding specified by Instruction itself.
    }

    if (!isValidEncoding(InstDef)) {
      ++NumEncodingsOmitted;
      continue;
    }

    unsigned EncodingID = Encodings.size();
    Encodings.emplace_back(InstDef, Inst);

    // This instruction is encoded the same on all HwModes.
    // According to user needs, add it to all, some, or only the default HwMode.
    handleHwModesUnrelatedEncodings(EncodingID, HwModeIDs,
                                    NamespacesWithHwModes);
  }

  for (const Record *EncodingDef :
       RK.getAllDerivedDefinitions("AdditionalEncoding")) {
    const Record *InstDef = EncodingDef->getValueAsDef("AliasOf");
    // TODO: Should probably give a warning in these cases.
    //   What's the point of specifying an additional encoding
    //   if it is invalid or if the instruction is not decodable?
    if (!isDecodableInstruction(InstDef)) {
      ++NumEncodingsLackingDisasm;
      continue;
    }
    if (!isValidEncoding(EncodingDef)) {
      ++NumEncodingsOmitted;
      continue;
    }
    unsigned EncodingID = Encodings.size();
    Encodings.emplace_back(EncodingDef, &Target.getInstruction(InstDef));
    EncodingIDsByHwMode[DefaultMode].push_back(EncodingID);
  }

  // Do some statistics.
  NumInstructions = Instructions.size();
  NumEncodingsSupported = Encodings.size();
  NumEncodings = NumEncodingsSupported + NumEncodingsOmitted;
}

DecoderEmitter::DecoderEmitter(const RecordKeeper &RK)
    : RK(RK), Target(RK), CGH(Target.getHwModes()) {
  Target.reverseBitsForLittleEndianEncoding();
  parseInstructionEncodings();
}

// Emits disassembler code for instruction decoding.
void DecoderEmitter::run(raw_ostream &o) const {
  formatted_raw_ostream OS(o);
  OS << R"(
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <assert.h>

namespace {

// InsnBitWidth is essentially a type trait used by the decoder emitter to query
// the supported bitwidth for a given type. But default, the value is 0, making
// it an invalid type for use as `InsnType` when instantiating the decoder.
// Individual targets are expected to provide specializations for these based
// on their usage.
template <typename T> constexpr uint32_t InsnBitWidth = 0;

)";

  // Do extra bookkeeping for variable-length encodings.
  bool IsVarLenInst = Target.hasVariableLengthEncodings();
  unsigned MaxInstLen = 0;
  if (IsVarLenInst) {
    std::vector<unsigned> InstrLen(Target.getInstructions().size(), 0);
    for (const InstructionEncoding &Encoding : Encodings) {
      MaxInstLen = std::max(MaxInstLen, Encoding.getBitWidth());
      InstrLen[Target.getInstrIntValue(Encoding.getInstruction()->TheDef)] =
          Encoding.getBitWidth();
    }

    // For variable instruction, we emit an instruction length table to let the
    // decoder know how long the instructions are. You can see example usage in
    // M68k's disassembler.
    emitInstrLenTable(OS, InstrLen);
  }

  // Map of (bitwidth, namespace, hwmode) tuple to encoding IDs.
  // Its organized as a nested map, with the (namespace, hwmode) as the key for
  // the inner map and bitwidth as the key for the outer map. We use std::map
  // for deterministic iteration order so that the code emitted is also
  // deterministic.
  using InnerKeyTy = std::pair<StringRef, unsigned>;
  using InnerMapTy = std::map<InnerKeyTy, std::vector<unsigned>>;
  std::map<unsigned, InnerMapTy> EncMap;

  for (const auto &[HwModeID, EncodingIDs] : EncodingIDsByHwMode) {
    for (unsigned EncodingID : EncodingIDs) {
      const InstructionEncoding &Encoding = Encodings[EncodingID];
      const unsigned BitWidth =
          IsVarLenInst ? MaxInstLen : Encoding.getBitWidth();
      StringRef DecoderNamespace = Encoding.getDecoderNamespace();
      EncMap[BitWidth][{DecoderNamespace, HwModeID}].push_back(EncodingID);
    }
  }

  // Variable length instructions use the same `APInt` type for all instructions
  // so we cannot specialize decoders based on instruction bitwidths (which
  // requires using different `InstType` for differet bitwidths for the correct
  // template specialization to kick in).
  if (IsVarLenInst && SpecializeDecodersPerBitwidth)
    PrintFatalError(
        "Cannot specialize decoders for variable length instuctions");

  // Entries in `EncMap` are already sorted by bitwidth. So bucketing per
  // bitwidth can be done on-the-fly as we iterate over the map.
  DecoderTableInfo TableInfo;
  DecoderTreeBuilder TreeBuilder(Target, Encodings, TableInfo);
  DecoderTableEmitter TableEmitter(Target, Encodings, OS);

  // Emit a table for each (namespace, hwmode, width) combination.
  // If `SpecializeDecodersPerBitwidth` is enabled, emit a decoder function
  // for each table.
  for (const auto &[BitWidth, BWMap] : EncMap) {
    for (const auto &[Key, EncodingIDs] : BWMap) {
      auto [DecoderNamespace, HwModeID] = Key;

      // Build a decoder for this (namespace, hwmode, width) combination.
      std::unique_ptr<DecoderTreeNode> Root =
          TreeBuilder.buildTree(EncodingIDs);

      SmallString<32> TableName("DecoderTable");
      TableName.append(DecoderNamespace);
      if (HwModeID != DefaultMode)
        TableName.append({"_", Target.getHwModes().getModeName(HwModeID)});
      TableName.append(std::to_string(BitWidth));

      // Serialize the tree.
      TableEmitter.emitTable(TableName, BitWidth, Root.get());
    }

    // Each BitWidth get's its own decoders and decoder function if
    // SpecializeDecodersPerBitwidth is enabled.
    if (SpecializeDecodersPerBitwidth) {
      emitDecoderFunction(OS, TableInfo.Decoders, BitWidth);
      TableInfo.Decoders.clear();
    }
  }

  // Emit the decoder function for the last bucket. This will also emit the
  // single decoder function if SpecializeDecodersPerBitwidth = false.
  if (!SpecializeDecodersPerBitwidth)
    emitDecoderFunction(OS, TableInfo.Decoders, 0);

  // Emit the predicate function.
  if (TableEmitter.hasCheckPredicate())
    emitPredicateFunction(OS, TableInfo.Predicates);

  // Emit the main entry point for the decoder, decodeInstruction().
  emitDecodeInstruction(OS, IsVarLenInst, TableEmitter.hasCheckPredicate(),
                        TableEmitter.hasSoftFail(),
                        TableEmitter.hasTryDecode());

  OS << "\n} // namespace\n";
}

void llvm::EmitDecoder(const RecordKeeper &RK, raw_ostream &OS) {
  DecoderEmitter(RK).run(OS);
}
