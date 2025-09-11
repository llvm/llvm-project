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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
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

static cl::opt<bool> LargeTable(
    "large-decoder-table",
    cl::desc("Use large decoder table format. This uses 24 bits for offset\n"
             "in the table instead of the default 16 bits."),
    cl::init(false), cl::cat(DisassemblerEmitterCat));

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

static unsigned getNumToSkipInBytes() { return LargeTable ? 3 : 2; }

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

typedef SmallSetVector<CachedHashString, 16> PredicateSet;
typedef SmallSetVector<CachedHashString, 16> DecoderSet;

class DecoderTable {
public:
  DecoderTable() { Data.reserve(16384); }

  void clear() { Data.clear(); }
  size_t size() const { return Data.size(); }
  const uint8_t *data() const { return Data.data(); }

  using const_iterator = std::vector<uint8_t>::const_iterator;
  const_iterator begin() const { return Data.begin(); }
  const_iterator end() const { return Data.end(); }

  /// Inserts a state machine opcode into the table.
  void insertOpcode(MCD::DecoderOps Opcode) { Data.push_back(Opcode); }

  /// Inserts a uint8 encoded value into the table.
  void insertUInt8(unsigned Value) {
    assert(isUInt<8>(Value));
    Data.push_back(Value);
  }

  /// Inserts a ULEB128 encoded value into the table.
  void insertULEB128(uint64_t Value) {
    // Encode and emit the value to filter against.
    uint8_t Buffer[16];
    unsigned Len = encodeULEB128(Value, Buffer);
    Data.insert(Data.end(), Buffer, Buffer + Len);
  }

  // Insert space for `NumToSkip` and return the position
  // in the table for patching.
  size_t insertNumToSkip() {
    size_t Size = Data.size();
    Data.insert(Data.end(), getNumToSkipInBytes(), 0);
    return Size;
  }

  void patchNumToSkip(size_t FixupIdx, uint32_t DestIdx) {
    // Calculate the distance from the byte following the fixup entry byte
    // to the destination. The Target is calculated from after the
    // `getNumToSkipInBytes()`-byte NumToSkip entry itself, so subtract
    // `getNumToSkipInBytes()` from the displacement here to account for that.
    assert(DestIdx >= FixupIdx + getNumToSkipInBytes() &&
           "Expecting a forward jump in the decoding table");
    uint32_t Delta = DestIdx - FixupIdx - getNumToSkipInBytes();
    if (!isUIntN(8 * getNumToSkipInBytes(), Delta))
      PrintFatalError(
          "disassembler decoding table too large, try --large-decoder-table");

    Data[FixupIdx] = static_cast<uint8_t>(Delta);
    Data[FixupIdx + 1] = static_cast<uint8_t>(Delta >> 8);
    if (getNumToSkipInBytes() == 3)
      Data[FixupIdx + 2] = static_cast<uint8_t>(Delta >> 16);
  }

private:
  std::vector<uint8_t> Data;
};

struct DecoderTableInfo {
  DecoderTable Table;
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

  // Emit the decoder state machine table. Returns a mask of MCD decoder ops
  // that were emitted.
  unsigned emitTable(formatted_raw_ostream &OS, DecoderTable &Table,
                     StringRef Namespace, unsigned HwModeID, unsigned BitWidth,
                     ArrayRef<unsigned> EncodingIDs) const;
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

class FilterChooser {
  // TODO: Unfriend by providing the necessary accessors.
  friend class DecoderTableBuilder;

  // Vector of encodings to choose our filter.
  ArrayRef<InstructionEncoding> Encodings;

  /// Encoding IDs for this filter chooser to work on.
  /// Sorted by non-decreasing encoding width.
  SmallVector<unsigned, 0> EncodingIDs;

  // Array of bit values passed down from our parent.
  // Set to all unknown for Parent == nullptr.
  KnownBits FilterBits;

  // Links to the FilterChooser above us in the decoding tree.
  const FilterChooser *Parent;

  /// If the selected filter matches multiple encodings, then this is the
  /// starting position and the width of the filtered range.
  unsigned StartBit;
  unsigned NumBits;

  /// If the selected filter matches multiple encodings, and there is
  /// *exactly one* encoding in which all bits are known in the filtered range,
  /// then this is the ID of that encoding.
  /// Also used when there is only one encoding.
  std::optional<unsigned> SingletonEncodingID;

  /// If the selected filter matches multiple encodings, and there is
  /// *at least one* encoding in which all bits are known in the filtered range,
  /// then this is the FilterChooser created for the subset of encodings that
  /// contain some unknown bits in the filtered range.
  std::unique_ptr<const FilterChooser> VariableFC;

  /// If the selected filter matches multiple encodings, and there is
  /// *more than one* encoding in which all bits are known in the filtered
  /// range, then this is a map of field values to FilterChoosers created for
  /// the subset of encodings sharing that field value.
  /// The "field value" here refers to the encoding bits in the filtered range.
  std::map<uint64_t, std::unique_ptr<const FilterChooser>> FilterChooserMap;

  /// Set to true if decoding conflict was encountered.
  bool HasConflict = false;

  struct Island {
    unsigned StartBit;
    unsigned NumBits;
    uint64_t FieldVal;
  };

public:
  /// Constructs a top-level filter chooser.
  FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                ArrayRef<unsigned> EncodingIDs)
      : Encodings(Encodings), EncodingIDs(EncodingIDs), Parent(nullptr) {
    // Sort encoding IDs once.
    stable_sort(this->EncodingIDs, LessEncodingIDByWidth(Encodings));
    // Filter width is the width of the smallest encoding.
    unsigned FilterWidth = Encodings[this->EncodingIDs.front()].getBitWidth();
    FilterBits = KnownBits(FilterWidth);
    doFilter();
  }

  /// Constructs an inferior filter chooser.
  FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                ArrayRef<unsigned> EncodingIDs, const KnownBits &FilterBits,
                const FilterChooser &Parent)
      : Encodings(Encodings), EncodingIDs(EncodingIDs), Parent(&Parent) {
    // Inferior filter choosers are created from sorted array of encoding IDs.
    assert(is_sorted(EncodingIDs, LessEncodingIDByWidth(Encodings)));
    assert(!FilterBits.hasConflict() && "Broken filter");
    // Filter width is the width of the smallest encoding.
    unsigned FilterWidth = Encodings[EncodingIDs.front()].getBitWidth();
    this->FilterBits = FilterBits.anyext(FilterWidth);
    doFilter();
  }

  FilterChooser(const FilterChooser &) = delete;
  void operator=(const FilterChooser &) = delete;

  /// Returns the width of the largest encoding.
  unsigned getMaxEncodingWidth() const {
    // The last encoding ID is the ID of an encoding with the largest width.
    return Encodings[EncodingIDs.back()].getBitWidth();
  }

  /// Returns true if any decoding conflicts were encountered.
  bool hasConflict() const { return HasConflict; }

private:
  /// Applies the given filter to the set of encodings this FilterChooser
  /// works with, creating inferior FilterChoosers as necessary.
  void applyFilter(const Filter &F);

  /// dumpStack - dumpStack traverses the filter chooser chain and calls
  /// dumpFilterArray on each filter chooser up to the top level one.
  void dumpStack(raw_ostream &OS, indent Indent, unsigned PadToWidth) const;

  bool isPositionFiltered(unsigned Idx) const {
    return FilterBits.Zero[Idx] || FilterBits.One[Idx];
  }

  // Calculates the island(s) needed to decode the instruction.
  // This returns a list of undecoded bits of an instructions, for example,
  // Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
  // decoded bits in order to verify that the instruction matches the Opcode.
  std::vector<Island> getIslands(const KnownBits &EncodingBits) const;

  /// Scans the well-known encoding bits of the encodings and, builds up a list
  /// of candidate filters, and then returns the best one, if any.
  std::unique_ptr<Filter> findBestFilter(ArrayRef<bitAttr_t> BitAttrs,
                                         bool AllowMixed,
                                         bool Greedy = true) const;

  std::unique_ptr<Filter> findBestFilter() const;

  // Decides on the best configuration of filter(s) to use in order to decode
  // the instructions.  A conflict of instructions may occur, in which case we
  // dump the conflict set to the standard error.
  void doFilter();

public:
  void dump() const;
};

class DecoderTableBuilder {
  const CodeGenTarget &Target;
  ArrayRef<InstructionEncoding> Encodings;
  DecoderTableInfo &TableInfo;

public:
  DecoderTableBuilder(const CodeGenTarget &Target,
                      ArrayRef<InstructionEncoding> Encodings,
                      DecoderTableInfo &TableInfo)
      : Target(Target), Encodings(Encodings), TableInfo(TableInfo) {}

  void buildTable(const FilterChooser &FC, unsigned BitWidth) const {
    // When specializing decoders per bit width, each decoder table will begin
    // with the bitwidth for that table.
    if (SpecializeDecodersPerBitwidth)
      TableInfo.Table.insertULEB128(BitWidth);
    emitTableEntries(FC);
  }

private:
  void emitBinaryParser(raw_ostream &OS, indent Indent,
                        const OperandInfo &OpInfo) const;

  void emitDecoder(raw_ostream &OS, indent Indent, unsigned EncodingID) const;

  unsigned getDecoderIndex(unsigned EncodingID) const;

  unsigned getPredicateIndex(StringRef P) const;

  bool emitPredicateMatchAux(const Init &Val, bool ParenIfBinOp,
                             raw_ostream &OS) const;

  bool emitPredicateMatch(raw_ostream &OS, unsigned EncodingID) const;

  bool doesOpcodeNeedPredicate(unsigned EncodingID) const;

  void emitPredicateTableEntry(unsigned EncodingID) const;

  void emitSoftFailTableEntry(unsigned EncodingID) const;

  void emitSingletonTableEntry(const FilterChooser &FC) const;

  void emitTableEntries(const FilterChooser &FC) const;
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

void FilterChooser::applyFilter(const Filter &F) {
  StartBit = F.StartBit;
  NumBits = F.NumBits;
  assert(FilterBits.extractBits(NumBits, StartBit).isUnknown());

  if (!F.VariableIDs.empty()) {
    // Delegates to an inferior filter chooser for further processing on this
    // group of instructions whose segment values are variable.
    VariableFC = std::make_unique<FilterChooser>(Encodings, F.VariableIDs,
                                                 FilterBits, *this);
    HasConflict |= VariableFC->HasConflict;
  }

  // Otherwise, create sub choosers.
  for (const auto &[FilterVal, InferiorEncodingIDs] : F.FilteredIDs) {
    // Create a new filter by inserting the field bits into the parent filter.
    APInt FieldBits(NumBits, FilterVal);
    KnownBits InferiorFilterBits = FilterBits;
    InferiorFilterBits.insertBits(KnownBits::makeConstant(FieldBits), StartBit);

    // Delegates to an inferior filter chooser for further processing on this
    // category of instructions.
    auto [It, _] = FilterChooserMap.try_emplace(
        FilterVal,
        std::make_unique<FilterChooser>(Encodings, InferiorEncodingIDs,
                                        InferiorFilterBits, *this));
    HasConflict |= It->second->HasConflict;
  }
}

// Returns the number of fanout produced by the filter.  More fanout implies
// the filter distinguishes more categories of instructions.
unsigned Filter::usefulness() const {
  return FilteredIDs.size() + VariableIDs.empty();
}

//////////////////////////////////
//                              //
// Filterchooser Implementation //
//                              //
//////////////////////////////////

// Emit the decoder state machine table. Returns a mask of MCD decoder ops
// that were emitted.
unsigned DecoderEmitter::emitTable(formatted_raw_ostream &OS,
                                   DecoderTable &Table, StringRef Namespace,
                                   unsigned HwModeID, unsigned BitWidth,
                                   ArrayRef<unsigned> EncodingIDs) const {
  // We'll need to be able to map from a decoded opcode into the corresponding
  // EncodingID for this specific combination of BitWidth and Namespace. This
  // is used below to index into Encodings.
  DenseMap<unsigned, unsigned> OpcodeToEncodingID;
  OpcodeToEncodingID.reserve(EncodingIDs.size());
  for (unsigned EncodingID : EncodingIDs) {
    const Record *InstDef = Encodings[EncodingID].getInstruction()->TheDef;
    OpcodeToEncodingID[Target.getInstrIntValue(InstDef)] = EncodingID;
  }

  OS << "static const uint8_t DecoderTable" << Namespace;
  if (HwModeID != DefaultMode)
    OS << '_' << Target.getHwModes().getModeName(HwModeID);
  OS << BitWidth << "[" << Table.size() << "] = {\n";

  // Emit ULEB128 encoded value to OS, returning the number of bytes emitted.
  auto emitULEB128 = [](DecoderTable::const_iterator &I,
                        formatted_raw_ostream &OS) {
    while (*I >= 128)
      OS << (unsigned)*I++ << ", ";
    OS << (unsigned)*I++ << ", ";
  };

  // Emit `getNumToSkipInBytes()`-byte numtoskip value to OS, returning the
  // NumToSkip value.
  auto emitNumToSkip = [](DecoderTable::const_iterator &I,
                          formatted_raw_ostream &OS) {
    uint8_t Byte = *I++;
    uint32_t NumToSkip = Byte;
    OS << (unsigned)Byte << ", ";
    Byte = *I++;
    OS << (unsigned)Byte << ", ";
    NumToSkip |= Byte << 8;
    if (getNumToSkipInBytes() == 3) {
      Byte = *I++;
      OS << (unsigned)(Byte) << ", ";
      NumToSkip |= Byte << 16;
    }
    return NumToSkip;
  };

  // FIXME: We may be able to use the NumToSkip values to recover
  // appropriate indentation levels.
  DecoderTable::const_iterator I = Table.begin();
  DecoderTable::const_iterator E = Table.end();
  const uint8_t *const EndPtr = Table.data() + Table.size();

  auto emitNumToSkipComment = [&](uint32_t NumToSkip, bool InComment = false) {
    uint32_t Index = ((I - Table.begin()) + NumToSkip);
    OS << (InComment ? ", " : "// ");
    OS << "Skip to: " << Index;
  };

  // The first entry when specializing decoders per bitwidth is the bitwidth.
  // This will be used for additional checks in `decodeInstruction`.
  if (SpecializeDecodersPerBitwidth) {
    OS << "/* 0  */";
    OS.PadToColumn(14);
    emitULEB128(I, OS);
    OS << " // Bitwidth " << BitWidth << '\n';
  }

  unsigned OpcodeMask = 0;

  while (I != E) {
    assert(I < E && "incomplete decode table entry!");

    uint64_t Pos = I - Table.begin();
    OS << "/* " << Pos << " */";
    OS.PadToColumn(12);

    const uint8_t DecoderOp = *I++;
    OpcodeMask |= (1 << DecoderOp);
    switch (DecoderOp) {
    default:
      PrintFatalError("Invalid decode table opcode: " + Twine((int)DecoderOp) +
                      " at index " + Twine(Pos));
    case MCD::OPC_Scope: {
      OS << "  MCD::OPC_Scope, ";
      uint32_t NumToSkip = emitNumToSkip(I, OS);
      emitNumToSkipComment(NumToSkip);
      OS << '\n';
      break;
    }
    case MCD::OPC_ExtractField: {
      OS << "  MCD::OPC_ExtractField, ";

      // ULEB128 encoded start value.
      const char *ErrMsg = nullptr;
      unsigned Start = decodeULEB128(&*I, nullptr, EndPtr, &ErrMsg);
      assert(ErrMsg == nullptr && "ULEB128 value too large!");
      emitULEB128(I, OS);

      unsigned Len = *I++;
      OS << Len << ",  // Inst{";
      if (Len > 1)
        OS << (Start + Len - 1) << "-";
      OS << Start << "} ...\n";
      break;
    }
    case MCD::OPC_FilterValueOrSkip: {
      OS << "  MCD::OPC_FilterValueOrSkip, ";
      // The filter value is ULEB128 encoded.
      emitULEB128(I, OS);
      uint32_t NumToSkip = emitNumToSkip(I, OS);
      emitNumToSkipComment(NumToSkip);
      OS << '\n';
      break;
    }
    case MCD::OPC_FilterValue: {
      OS << "  MCD::OPC_FilterValue, ";
      // The filter value is ULEB128 encoded.
      emitULEB128(I, OS);
      OS << '\n';
      break;
    }
    case MCD::OPC_CheckField: {
      OS << "  MCD::OPC_CheckField, ";
      // ULEB128 encoded start value.
      emitULEB128(I, OS);
      // 8-bit length.
      unsigned Len = *I++;
      OS << Len << ", ";
      // ULEB128 encoded field value.
      emitULEB128(I, OS);
      OS << '\n';
      break;
    }
    case MCD::OPC_CheckPredicate: {
      OS << "  MCD::OPC_CheckPredicate, ";
      emitULEB128(I, OS);
      OS << '\n';
      break;
    }
    case MCD::OPC_Decode:
    case MCD::OPC_TryDecode: {
      bool IsTry = DecoderOp == MCD::OPC_TryDecode;
      // Decode the Opcode value.
      const char *ErrMsg = nullptr;
      unsigned Opc = decodeULEB128(&*I, nullptr, EndPtr, &ErrMsg);
      assert(ErrMsg == nullptr && "ULEB128 value too large!");

      OS << "  MCD::OPC_" << (IsTry ? "Try" : "") << "Decode, ";
      emitULEB128(I, OS);

      // Decoder index.
      unsigned DecodeIdx = decodeULEB128(&*I, nullptr, EndPtr, &ErrMsg);
      assert(ErrMsg == nullptr && "ULEB128 value too large!");
      emitULEB128(I, OS);

      auto EncI = OpcodeToEncodingID.find(Opc);
      assert(EncI != OpcodeToEncodingID.end() && "no encoding entry");
      auto EncodingID = EncI->second;

      if (!IsTry) {
        OS << "// Opcode: " << Encodings[EncodingID].getName()
           << ", DecodeIdx: " << DecodeIdx << '\n';
        break;
      }
      OS << '\n';
      break;
    }
    case MCD::OPC_SoftFail: {
      OS << "  MCD::OPC_SoftFail, ";
      // Decode the positive mask.
      const char *ErrMsg = nullptr;
      uint64_t PositiveMask = decodeULEB128(&*I, nullptr, EndPtr, &ErrMsg);
      assert(ErrMsg == nullptr && "ULEB128 value too large!");
      emitULEB128(I, OS);

      // Decode the negative mask.
      uint64_t NegativeMask = decodeULEB128(&*I, nullptr, EndPtr, &ErrMsg);
      assert(ErrMsg == nullptr && "ULEB128 value too large!");
      emitULEB128(I, OS);
      OS << "// +ve mask: 0x";
      OS.write_hex(PositiveMask);
      OS << ", -ve mask: 0x";
      OS.write_hex(NegativeMask);
      OS << '\n';
      break;
    }
    }
  }
  OS << "};\n\n";

  return OpcodeMask;
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
void FilterChooser::dumpStack(raw_ostream &OS, indent Indent,
                              unsigned PadToWidth) const {
  if (Parent)
    Parent->dumpStack(OS, Indent, PadToWidth);
  assert(PadToWidth >= FilterBits.getBitWidth());
  OS << Indent << indent(PadToWidth - FilterBits.getBitWidth());
  printKnownBits(OS, FilterBits, '.');
  OS << '\n';
}

// Calculates the island(s) needed to decode the instruction.
// This returns a list of undecoded bits of an instructions, for example,
// Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
// decoded bits in order to verify that the instruction matches the Opcode.
std::vector<FilterChooser::Island>
FilterChooser::getIslands(const KnownBits &EncodingBits) const {
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
    bool Filtered = isPositionFiltered(i);
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

void DecoderTableBuilder::emitBinaryParser(raw_ostream &OS, indent Indent,
                                           const OperandInfo &OpInfo) const {
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

void DecoderTableBuilder::emitDecoder(raw_ostream &OS, indent Indent,
                                      unsigned EncodingID) const {
  const InstructionEncoding &Encoding = Encodings[EncodingID];

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

unsigned DecoderTableBuilder::getDecoderIndex(unsigned EncodingID) const {
  // Build up the predicate string.
  SmallString<256> Decoder;
  // FIXME: emitDecoder() function can take a buffer directly rather than
  // a stream.
  raw_svector_ostream S(Decoder);
  indent Indent(UseFnTableInDecodeToMCInst ? 2 : 4);
  emitDecoder(S, Indent, EncodingID);

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
bool DecoderTableBuilder::emitPredicateMatchAux(const Init &Val,
                                                bool ParenIfBinOp,
                                                raw_ostream &OS) const {
  if (const auto *D = dyn_cast<DefInit>(&Val)) {
    if (!D->getDef()->isSubClassOf("SubtargetFeature"))
      return true;
    OS << "Bits[" << Target.getName() << "::" << D->getAsString() << "]";
    return false;
  }
  if (const auto *D = dyn_cast<DagInit>(&Val)) {
    std::string Op = D->getOperator()->getAsString();
    if (Op == "not" && D->getNumArgs() == 1) {
      OS << '!';
      return emitPredicateMatchAux(*D->getArg(0), true, OS);
    }
    if ((Op == "any_of" || Op == "all_of") && D->getNumArgs() > 0) {
      bool Paren = D->getNumArgs() > 1 && std::exchange(ParenIfBinOp, true);
      if (Paren)
        OS << '(';
      ListSeparator LS(Op == "any_of" ? " || " : " && ");
      for (auto *Arg : D->getArgs()) {
        OS << LS;
        if (emitPredicateMatchAux(*Arg, ParenIfBinOp, OS))
          return true;
      }
      if (Paren)
        OS << ')';
      return false;
    }
  }
  return true;
}

bool DecoderTableBuilder::emitPredicateMatch(raw_ostream &OS,
                                             unsigned EncodingID) const {
  const ListInit *Predicates =
      Encodings[EncodingID].getRecord()->getValueAsListInit("Predicates");
  bool IsFirstEmission = true;
  for (unsigned i = 0; i < Predicates->size(); ++i) {
    const Record *Pred = Predicates->getElementAsRecord(i);
    if (!Pred->getValue("AssemblerMatcherPredicate"))
      continue;

    if (!isa<DagInit>(Pred->getValue("AssemblerCondDag")->getValue()))
      continue;

    if (!IsFirstEmission)
      OS << " && ";
    if (emitPredicateMatchAux(*Pred->getValueAsDag("AssemblerCondDag"),
                              Predicates->size() > 1, OS))
      PrintFatalError(Pred->getLoc(), "Invalid AssemblerCondDag!");
    IsFirstEmission = false;
  }
  return !Predicates->empty();
}

bool DecoderTableBuilder::doesOpcodeNeedPredicate(unsigned EncodingID) const {
  const ListInit *Predicates =
      Encodings[EncodingID].getRecord()->getValueAsListInit("Predicates");
  for (unsigned i = 0; i < Predicates->size(); ++i) {
    const Record *Pred = Predicates->getElementAsRecord(i);
    if (!Pred->getValue("AssemblerMatcherPredicate"))
      continue;

    if (isa<DagInit>(Pred->getValue("AssemblerCondDag")->getValue()))
      return true;
  }
  return false;
}

unsigned DecoderTableBuilder::getPredicateIndex(StringRef Predicate) const {
  // Using the full predicate string as the key value here is a bit
  // heavyweight, but is effective. If the string comparisons become a
  // performance concern, we can implement a mangling of the predicate
  // data easily enough with a map back to the actual string. That's
  // overkill for now, though.

  // Make sure the predicate is in the table.
  TableInfo.Predicates.insert(CachedHashString(Predicate));
  // Now figure out the index for when we write out the table.
  PredicateSet::const_iterator P = find(TableInfo.Predicates, Predicate);
  return (unsigned)(P - TableInfo.Predicates.begin());
}

void DecoderTableBuilder::emitPredicateTableEntry(unsigned EncodingID) const {
  if (!doesOpcodeNeedPredicate(EncodingID))
    return;

  // Build up the predicate string.
  SmallString<256> Predicate;
  // FIXME: emitPredicateMatch() functions can take a buffer directly rather
  // than a stream.
  raw_svector_ostream PS(Predicate);
  emitPredicateMatch(PS, EncodingID);

  // Figure out the index into the predicate table for the predicate just
  // computed.
  unsigned PIdx = getPredicateIndex(PS.str());

  TableInfo.Table.insertOpcode(MCD::OPC_CheckPredicate);
  TableInfo.Table.insertULEB128(PIdx);
}

void DecoderTableBuilder::emitSoftFailTableEntry(unsigned EncodingID) const {
  const InstructionEncoding &Encoding = Encodings[EncodingID];
  const KnownBits &InstBits = Encoding.getInstBits();
  const APInt &SoftFailMask = Encoding.getSoftFailMask();

  if (SoftFailMask.isZero())
    return;

  APInt PositiveMask = InstBits.Zero & SoftFailMask;
  APInt NegativeMask = InstBits.One & SoftFailMask;

  TableInfo.Table.insertOpcode(MCD::OPC_SoftFail);
  TableInfo.Table.insertULEB128(PositiveMask.getZExtValue());
  TableInfo.Table.insertULEB128(NegativeMask.getZExtValue());
}

// Emits table entries to decode the singleton.
void DecoderTableBuilder::emitSingletonTableEntry(
    const FilterChooser &FC) const {
  unsigned EncodingID = *FC.SingletonEncodingID;
  const InstructionEncoding &Encoding = Encodings[EncodingID];
  KnownBits EncodingBits = Encoding.getMandatoryBits();

  // Look for islands of undecoded bits of the singleton.
  std::vector<FilterChooser::Island> Islands = FC.getIslands(EncodingBits);

  // Emit the predicate table entry if one is needed.
  emitPredicateTableEntry(EncodingID);

  // Check any additional encoding fields needed.
  for (const FilterChooser::Island &Ilnd : reverse(Islands)) {
    TableInfo.Table.insertOpcode(MCD::OPC_CheckField);
    TableInfo.Table.insertULEB128(Ilnd.StartBit);
    TableInfo.Table.insertUInt8(Ilnd.NumBits);
    TableInfo.Table.insertULEB128(Ilnd.FieldVal);
  }

  // Check for soft failure of the match.
  emitSoftFailTableEntry(EncodingID);

  unsigned DIdx = getDecoderIndex(EncodingID);

  // Produce OPC_Decode or OPC_TryDecode opcode based on the information
  // whether the instruction decoder is complete or not. If it is complete
  // then it handles all possible values of remaining variable/unfiltered bits
  // and for any value can determine if the bitpattern is a valid instruction
  // or not. This means OPC_Decode will be the final step in the decoding
  // process. If it is not complete, then the Fail return code from the
  // decoder method indicates that additional processing should be done to see
  // if there is any other instruction that also matches the bitpattern and
  // can decode it.
  const MCD::DecoderOps DecoderOp =
      Encoding.hasCompleteDecoder() ? MCD::OPC_Decode : MCD::OPC_TryDecode;
  TableInfo.Table.insertOpcode(DecoderOp);
  const Record *InstDef = Encodings[EncodingID].getInstruction()->TheDef;
  TableInfo.Table.insertULEB128(Target.getInstrIntValue(InstDef));
  TableInfo.Table.insertULEB128(DIdx);
}

std::unique_ptr<Filter>
FilterChooser::findBestFilter(ArrayRef<bitAttr_t> BitAttrs, bool AllowMixed,
                              bool Greedy) const {
  assert(EncodingIDs.size() >= 2 && "Nothing to filter");

  // Heuristics.  See also doFilter()'s "Heuristics" comment when num of
  // instructions is 3.
  if (AllowMixed && !Greedy) {
    assert(EncodingIDs.size() == 3);

    for (unsigned EncodingID : EncodingIDs) {
      const InstructionEncoding &Encoding = Encodings[EncodingID];
      KnownBits EncodingBits = Encoding.getMandatoryBits();

      // Look for islands of undecoded bits of any instruction.
      std::vector<Island> Islands = getIslands(EncodingBits);
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

  auto addCandidateFilter = [&](unsigned StartBit, unsigned EndBit) {
    Filters.push_back(std::make_unique<Filter>(Encodings, EncodingIDs, StartBit,
                                               EndBit - StartBit));
  };

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
      if (!AllowMixed && bitAttr != ATTR_ALL_SET)
        addCandidateFilter(StartBit, BitIndex);
      switch (bitAttr) {
      case ATTR_FILTERED:
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        break;
      case ATTR_ALL_UNSET:
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        StartBit = BitIndex;
        RA = ATTR_MIXED;
        break;
      default:
        llvm_unreachable("Unexpected bitAttr!");
      }
      break;
    case ATTR_MIXED:
      if (AllowMixed && bitAttr != ATTR_MIXED)
        addCandidateFilter(StartBit, BitIndex);
      switch (bitAttr) {
      case ATTR_FILTERED:
        StartBit = BitIndex;
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
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
    if (!AllowMixed)
      addCandidateFilter(StartBit, FilterWidth);
    break;
  case ATTR_ALL_UNSET:
    break;
  case ATTR_MIXED:
    if (AllowMixed)
      addCandidateFilter(StartBit, FilterWidth);
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

std::unique_ptr<Filter> FilterChooser::findBestFilter() const {
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
    if (isPositionFiltered(BitIndex))
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
  if (std::unique_ptr<Filter> F =
          findBestFilter(BitAttrs, /*AllowMixed=*/false))
    return F;

  // Then regions of mixed bits (both known and unitialized bit values allowed).
  if (std::unique_ptr<Filter> F = findBestFilter(BitAttrs, /*AllowMixed=*/true))
    return F;

  // Heuristics to cope with conflict set {t2CMPrs, t2SUBSrr, t2SUBSrs} where
  // no single instruction for the maximum ATTR_MIXED region Inst{14-4} has a
  // well-known encoding pattern.  In such case, we backtrack and scan for the
  // the very first consecutive ATTR_ALL_SET region and assign a filter to it.
  if (EncodingIDs.size() == 3) {
    if (std::unique_ptr<Filter> F =
            findBestFilter(BitAttrs, /*AllowMixed=*/true, /*Greedy=*/false))
      return F;
  }

  // There is a conflict we could not resolve.
  return nullptr;
}

// Decides on the best configuration of filter(s) to use in order to decode
// the instructions.  A conflict of instructions may occur, in which case we
// dump the conflict set to the standard error.
void FilterChooser::doFilter() {
  assert(!EncodingIDs.empty() && "FilterChooser created with no instructions");

  // No filter needed.
  if (EncodingIDs.size() == 1) {
    SingletonEncodingID = EncodingIDs.front();
    return;
  }

  std::unique_ptr<Filter> BestFilter = findBestFilter();
  if (BestFilter) {
    applyFilter(*BestFilter);
    return;
  }

  // Print out useful conflict information for postmortem analysis.
  errs() << "Decoding Conflict:\n";
  dump();
  HasConflict = true;
}

void FilterChooser::dump() const {
  indent Indent(4);
  // Helps to keep the output right-justified.
  unsigned PadToWidth = getMaxEncodingWidth();

  // Dump filter stack.
  dumpStack(errs(), Indent, PadToWidth);

  // Dump encodings.
  for (unsigned EncodingID : EncodingIDs) {
    const InstructionEncoding &Encoding = Encodings[EncodingID];
    errs() << Indent << indent(PadToWidth - Encoding.getBitWidth());
    printKnownBits(errs(), Encoding.getMandatoryBits(), '_');
    errs() << "  " << Encoding.getName() << '\n';
  }
}

void DecoderTableBuilder::emitTableEntries(const FilterChooser &FC) const {
  DecoderTable &Table = TableInfo.Table;

  // If there are other encodings that could match if those with all bits
  // known don't, enter a scope so that they have a chance.
  size_t FixupLoc = 0;
  if (FC.VariableFC) {
    Table.insertOpcode(MCD::OPC_Scope);
    FixupLoc = Table.insertNumToSkip();
  }

  if (FC.SingletonEncodingID) {
    assert(FC.FilterChooserMap.empty());
    // There is only one encoding in which all bits in the filtered range are
    // fully defined, but we still need to check if the remaining (unfiltered)
    // bits are valid for this encoding. We also need to check predicates etc.
    emitSingletonTableEntry(FC);
  } else if (FC.FilterChooserMap.size() == 1) {
    // If there is only one possible field value, emit a combined OPC_CheckField
    // instead of OPC_ExtractField + OPC_FilterValue.
    const auto &[FilterVal, Delegate] = *FC.FilterChooserMap.begin();
    Table.insertOpcode(MCD::OPC_CheckField);
    Table.insertULEB128(FC.StartBit);
    Table.insertUInt8(FC.NumBits);
    Table.insertULEB128(FilterVal);

    // Emit table entries for the only case.
    emitTableEntries(*Delegate);
  } else {
    // The general case: emit a switch over the field value.
    Table.insertOpcode(MCD::OPC_ExtractField);
    Table.insertULEB128(FC.StartBit);
    Table.insertUInt8(FC.NumBits);

    // Emit switch cases for all but the last element.
    for (const auto &[FilterVal, Delegate] : drop_end(FC.FilterChooserMap)) {
      Table.insertOpcode(MCD::OPC_FilterValueOrSkip);
      Table.insertULEB128(FilterVal);
      size_t FixupPos = Table.insertNumToSkip();

      // Emit table entries for this case.
      emitTableEntries(*Delegate);

      // Patch the previous FilterValueOrSkip to fall through to the next case.
      Table.patchNumToSkip(FixupPos, Table.size());
    }

    // Emit a switch case for the last element. It never falls through;
    // if it doesn't match, we leave the current scope.
    const auto &[FilterVal, Delegate] = *FC.FilterChooserMap.rbegin();
    Table.insertOpcode(MCD::OPC_FilterValue);
    Table.insertULEB128(FilterVal);

    // Emit table entries for the last case.
    emitTableEntries(*Delegate);
  }

  if (FC.VariableFC) {
    Table.patchNumToSkip(FixupLoc, Table.size());
    emitTableEntries(*FC.VariableFC);
  }
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
                                  unsigned OpcodeMask) {
  const bool HasTryDecode = OpcodeMask & (1 << MCD::OPC_TryDecode);
  const bool HasCheckPredicate = OpcodeMask & (1 << MCD::OPC_CheckPredicate);
  const bool HasSoftFail = OpcodeMask & (1 << MCD::OPC_SoftFail);

  OS << R"(
static unsigned decodeNumToSkip(const uint8_t *&Ptr) {
  unsigned NumToSkip = *Ptr++;
  NumToSkip |= (*Ptr++) << 8;
)";
  if (getNumToSkipInBytes() == 3)
    OS << "  NumToSkip |= (*Ptr++) << 16;\n";
  OS << R"(  return NumToSkip;
}

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
  uint64_t CurFieldValue = 0;
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
      unsigned NumToSkip = decodeNumToSkip(Ptr);
      const uint8_t *SkipTo = Ptr + NumToSkip;
      ScopeStack.push_back(SkipTo);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Scope(" << SkipTo - DecodeTable
                        << ")\n");
      break;
    }
    case MCD::OPC_ExtractField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(Ptr);
      unsigned Len = *Ptr++;)";
  if (IsVarLenInst)
    OS << "\n      makeUp(insn, Start + Len);";
  OS << R"(
      CurFieldValue = fieldFromInstruction(insn, Start, Len);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_ExtractField(" << Start << ", "
                   << Len << "): " << CurFieldValue << "\n");
      break;
    }
    case MCD::OPC_FilterValueOrSkip: {
      // Decode the field value.
      uint64_t Val = decodeULEB128AndIncUnsafe(Ptr);
      bool Failed = Val != CurFieldValue;
      unsigned NumToSkip = decodeNumToSkip(Ptr);
      const uint8_t *SkipTo = Ptr + NumToSkip;

      LLVM_DEBUG(dbgs() << Loc << ": OPC_FilterValueOrSkip(" << Val << ", "
                        << SkipTo - DecodeTable << ") "
                        << (Failed ? "FAIL, " : "PASS\n"));

      if (Failed) {
        Ptr = SkipTo;
        LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
      }
      break;
    }
    case MCD::OPC_FilterValue: {
      // Decode the field value.
      uint64_t Val = decodeULEB128AndIncUnsafe(Ptr);
      bool Failed = Val != CurFieldValue;

      LLVM_DEBUG(dbgs() << Loc << ": OPC_FilterValue(" << Val << ") "
                        << (Failed ? "FAIL, " : "PASS\n"));

      if (Failed) {
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
  DecoderTableBuilder TableBuilder(Target, Encodings, TableInfo);
  unsigned OpcodeMask = 0;

  bool HasConflict = false;
  for (const auto &[BitWidth, BWMap] : EncMap) {
    for (const auto &[Key, EncodingIDs] : BWMap) {
      auto [DecoderNamespace, HwModeID] = Key;

      // Emit the decoder for this (namespace, hwmode, width) combination.
      FilterChooser FC(Encodings, EncodingIDs);
      HasConflict |= FC.hasConflict();
      // Skip emitting table entries if a conflict has been detected.
      if (HasConflict)
        continue;

      // The decode table is cleared for each top level decoder function. The
      // predicates and decoders themselves, however, are shared across
      // different decoders to give more opportunities for uniqueing.
      //  - If `SpecializeDecodersPerBitwidth` is enabled, decoders are shared
      //    across all decoder tables for a given bitwidth, else they are shared
      //    across all decoder tables.
      //  - predicates are shared across all decoder tables.
      TableInfo.Table.clear();
      TableBuilder.buildTable(FC, BitWidth);

      // Print the table to the output stream.
      OpcodeMask |= emitTable(OS, TableInfo.Table, DecoderNamespace, HwModeID,
                              BitWidth, EncodingIDs);
    }

    // Each BitWidth get's its own decoders and decoder function if
    // SpecializeDecodersPerBitwidth is enabled.
    if (SpecializeDecodersPerBitwidth) {
      emitDecoderFunction(OS, TableInfo.Decoders, BitWidth);
      TableInfo.Decoders.clear();
    }
  }

  if (HasConflict)
    PrintFatalError("Decoding conflict encountered");

  // Emit the decoder function for the last bucket. This will also emit the
  // single decoder function if SpecializeDecodersPerBitwidth = false.
  if (!SpecializeDecodersPerBitwidth)
    emitDecoderFunction(OS, TableInfo.Decoders, 0);

  const bool HasCheckPredicate = OpcodeMask & (1 << MCD::OPC_CheckPredicate);

  // Emit the predicate function.
  if (HasCheckPredicate)
    emitPredicateFunction(OS, TableInfo.Predicates);

  // Emit the main entry point for the decoder, decodeInstruction().
  emitDecodeInstruction(OS, IsVarLenInst, OpcodeMask);

  OS << "\n} // namespace\n";
}

void llvm::EmitDecoder(const RecordKeeper &RK, raw_ostream &OS) {
  DecoderEmitter(RK).run(OS);
}
