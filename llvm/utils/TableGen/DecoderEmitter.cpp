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

struct EncodingField {
  unsigned Base, Width, Offset;
  EncodingField(unsigned B, unsigned W, unsigned O)
      : Base(B), Width(W), Offset(O) {}
};

struct OperandInfo {
  std::vector<EncodingField> Fields;
  std::string Decoder;
  bool HasCompleteDecoder;
  uint64_t InitValue = 0;

  OperandInfo(std::string D, bool HCD) : Decoder(D), HasCompleteDecoder(HCD) {}

  void addField(unsigned Base, unsigned Width, unsigned Offset) {
    Fields.push_back(EncodingField(Base, Width, Offset));
  }

  unsigned numFields() const { return Fields.size(); }

  typedef std::vector<EncodingField>::const_iterator const_iterator;

  const_iterator begin() const { return Fields.begin(); }
  const_iterator end() const { return Fields.end(); }
};

/// Represents a parsed InstructionEncoding record or a record derived from it.
class InstructionEncoding {
  /// The Record this encoding originates from.
  const Record *EncodingDef;

  /// The instruction this encoding is for.
  const CodeGenInstruction *Inst;

  /// The name of this encoding (for debugging purposes).
  std::string Name;

  /// Known bits of this encoding. This is the value of the `Inst` field
  /// with any variable references replaced with '?'.
  KnownBits InstBits;

  /// Mask of bits that should be considered unknown during decoding.
  /// This is the value of the `SoftFail` field.
  APInt SoftFailBits;

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

  /// Returns the size of this encoding, in bits.
  unsigned getBitWidth() const { return InstBits.getBitWidth(); }

  /// Returns the known bits of this encoding.
  const KnownBits &getInstBits() const { return InstBits; }

  /// Returns a mask of bits that should be considered unknown during decoding.
  const APInt &getSoftFailBits() const { return SoftFailBits; }

  /// Returns the known bits of this encoding that must match for
  /// successful decoding.
  KnownBits getMandatoryBits() const {
    KnownBits EncodingBits = InstBits;
    // Mark all bits that are allowed to change according to SoftFail mask
    // as unknown.
    EncodingBits.Zero &= ~SoftFailBits;
    EncodingBits.One &= ~SoftFailBits;
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
  void parseFixedLenEncoding(const BitsInit &Bits);

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

typedef std::vector<uint32_t> FixupList;
typedef std::vector<FixupList> FixupScopeList;
typedef SmallSetVector<CachedHashString, 16> PredicateSet;
typedef SmallSetVector<CachedHashString, 16> DecoderSet;

class DecoderTable {
public:
  DecoderTable() { Data.reserve(16384); }

  void clear() { Data.clear(); }
  void push_back(uint8_t Item) { Data.push_back(Item); }
  size_t size() const { return Data.size(); }
  const uint8_t *data() const { return Data.data(); }

  using const_iterator = std::vector<uint8_t>::const_iterator;
  const_iterator begin() const { return Data.begin(); }
  const_iterator end() const { return Data.end(); }

  // Insert a ULEB128 encoded value into the table.
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
  FixupScopeList FixupStack;
  PredicateSet Predicates;
  DecoderSet Decoders;

  bool isOutermostScope() const { return FixupStack.size() == 1; }

  void pushScope() { FixupStack.emplace_back(); }

  void popScope() {
    // Resolve any remaining fixups in the current scope before popping it.
    // All fixups resolve to the current location.
    uint32_t DestIdx = Table.size();
    for (uint32_t FixupIdx : FixupStack.back())
      Table.patchNumToSkip(FixupIdx, DestIdx);
    FixupStack.pop_back();
  }
};

using NamespacesHwModesMap = std::map<std::string, std::set<unsigned>>;

class DecoderEmitter {
  const RecordKeeper &RK;
  CodeGenTarget Target;
  const CodeGenHwModes &CGH;

  /// All parsed encodings.
  std::vector<InstructionEncoding> Encodings;

  /// Encodings IDs for each HwMode. An ID is an index into Encodings.
  SmallDenseMap<unsigned, std::vector<unsigned>> EncodingIDsByHwMode;

public:
  DecoderEmitter(const RecordKeeper &RK, StringRef PredicateNamespace);

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
                           DecoderSet &Decoders) const;

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

public:
  StringRef PredicateNamespace;
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

  bool hasSingleFilteredID() const {
    return FilteredIDs.size() == 1 && FilteredIDs.begin()->second.size() == 1;
  }

  unsigned getSingletonEncodingID() const {
    assert(hasSingleFilteredID());
    return FilteredIDs.begin()->second.front();
  }

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

  /// Some targets (ARM) specify more encoding bits in Inst that Size allows.
  /// This field allows us to ignore the extra bits.
  unsigned MaxFilterWidth;

  // Parent emitter
  const DecoderEmitter *Emitter;

  /// If the selected filter matches multiple encodings, then this is the
  /// starting position and the width of the filtered range.
  unsigned StartBit;
  unsigned NumBits;

  /// If the selected filter matches multiple encodings, and there is
  /// *exactly one* encoding in which all bits are known in the filtered range,
  /// then this is the ID of that encoding.
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

  struct Island {
    unsigned StartBit;
    unsigned NumBits;
    uint64_t FieldVal;
  };

public:
  /// Constructs a top-level filter chooser.
  FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                ArrayRef<unsigned> EncodingIDs, unsigned MaxFilterWidth,
                const DecoderEmitter *E)
      : Encodings(Encodings), EncodingIDs(EncodingIDs), Parent(nullptr),
        MaxFilterWidth(MaxFilterWidth), Emitter(E) {
    // Sort encoding IDs once.
    stable_sort(this->EncodingIDs, LessEncodingIDByWidth(Encodings));
    // Filter width is the width of the smallest encoding.
    unsigned FilterWidth = Encodings[this->EncodingIDs.front()].getBitWidth();
    // Cap it as necessary.
    FilterWidth = std::min(FilterWidth, MaxFilterWidth);
    FilterBits = KnownBits(FilterWidth);
    doFilter();
  }

  /// Constructs an inferior filter chooser.
  FilterChooser(ArrayRef<InstructionEncoding> Encodings,
                ArrayRef<unsigned> EncodingIDs, const KnownBits &FilterBits,
                const FilterChooser &Parent)
      : Encodings(Encodings), EncodingIDs(EncodingIDs), Parent(&Parent),
        MaxFilterWidth(Parent.MaxFilterWidth), Emitter(Parent.Emitter) {
    // Inferior filter choosers are created from sorted array of encoding IDs.
    assert(is_sorted(EncodingIDs, LessEncodingIDByWidth(Encodings)));
    assert(!FilterBits.hasConflict() && "Broken filter");
    // Filter width is the width of the smallest encoding.
    unsigned FilterWidth = Encodings[EncodingIDs.front()].getBitWidth();
    // Cap it as necessary.
    FilterWidth = std::min(FilterWidth, MaxFilterWidth);
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

  // Emits code to check the Predicates member of an instruction are true.
  // Returns true if predicate matches were emitted, false otherwise.
  bool emitPredicateMatch(raw_ostream &OS, unsigned EncodingID) const;
  bool emitPredicateMatchAux(const Init &Val, bool ParenIfBinOp,
                             raw_ostream &OS) const;

  bool doesOpcodeNeedPredicate(unsigned EncodingID) const;
  unsigned getPredicateIndex(DecoderTableInfo &TableInfo, StringRef P) const;
  void emitPredicateTableEntry(DecoderTableInfo &TableInfo,
                               unsigned EncodingID) const;

  void emitSoftFailTableEntry(DecoderTableInfo &TableInfo,
                              unsigned EncodingID) const;

  // Emits table entries to decode the singleton.
  void emitSingletonTableEntry(DecoderTableInfo &TableInfo,
                               unsigned EncodingID) const;

  // Emits code to decode the singleton, and then to decode the rest.
  void emitSingletonTableEntry(DecoderTableInfo &TableInfo) const;

  // Emit table entries to decode instructions given a segment or segments of
  // bits.
  void emitTableEntry(DecoderTableInfo &TableInfo) const;

  void emitBinaryParser(raw_ostream &OS, indent Indent,
                        const OperandInfo &OpInfo) const;

  void emitDecoder(raw_ostream &OS, indent Indent, unsigned EncodingID) const;
  unsigned getDecoderIndex(DecoderSet &Decoders, unsigned EncodingID) const;

  // reportRegion is a helper function for filterProcessor to mark a region as
  // eligible for use as a filter region.
  void reportRegion(std::vector<std::unique_ptr<Filter>> &Filters, bitAttr_t RA,
                    unsigned StartBit, unsigned BitIndex,
                    bool AllowMixed) const;

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
  // emitTableEntries - Emit state machine entries to decode our share of
  // instructions.
  void emitTableEntries(DecoderTableInfo &TableInfo) const;

  void dump() const;
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
  }

  // No need to recurse for a singleton filtered instruction.
  // See also Filter::emit*().
  if (F.hasSingleFilteredID()) {
    SingletonEncodingID = F.getSingletonEncodingID();
    assert(VariableFC && "Shouldn't have created a filter for one encoding!");
    return;
  }

  // Otherwise, create sub choosers.
  for (const auto &[FilterVal, InferiorEncodingIDs] : F.FilteredIDs) {
    // Create a new filter by inserting the field bits into the parent filter.
    APInt FieldBits(NumBits, FilterVal);
    KnownBits InferiorFilterBits = FilterBits;
    InferiorFilterBits.insertBits(KnownBits::makeConstant(FieldBits), StartBit);

    // Delegates to an inferior filter chooser for further processing on this
    // category of instructions.
    FilterChooserMap.try_emplace(FilterVal, std::make_unique<FilterChooser>(
                                                Encodings, InferiorEncodingIDs,
                                                InferiorFilterBits, *this));
  }
}

// Emit table entries to decode instructions given a segment or segments
// of bits.
void FilterChooser::emitTableEntry(DecoderTableInfo &TableInfo) const {
  assert(isUInt<8>(NumBits) && "NumBits overflowed uint8 table entry!");
  TableInfo.Table.push_back(MCD::OPC_ExtractField);

  TableInfo.Table.insertULEB128(StartBit);
  TableInfo.Table.push_back(NumBits);

  // If VariableFC is present, we need to add a new scope for this filter.
  // Otherwise, we can skip adding a new scope and any patching added will
  // automatically be added to the enclosing scope.
  const uint64_t LastFilter = FilterChooserMap.rbegin()->first;
  if (VariableFC)
    TableInfo.FixupStack.emplace_back();

  DecoderTable &Table = TableInfo.Table;

  size_t PrevFilter = 0;
  for (const auto &[FilterVal, Delegate] : FilterChooserMap) {
    // The last filtervalue emitted can be OPC_FilterValue if we are at
    // outermost scope.
    const uint8_t DecoderOp =
        FilterVal == LastFilter && TableInfo.isOutermostScope()
            ? MCD::OPC_FilterValueOrFail
            : MCD::OPC_FilterValue;
    Table.push_back(DecoderOp);
    Table.insertULEB128(FilterVal);
    if (DecoderOp == MCD::OPC_FilterValue) {
      // Reserve space for the NumToSkip entry. We'll backpatch the value later.
      PrevFilter = Table.insertNumToSkip();
    } else {
      PrevFilter = 0;
    }

    // We arrive at a category of instructions with the same segment value.
    // Now delegate to the sub filter chooser for further decodings.
    // The case may fallthrough, which happens if the remaining well-known
    // encoding bits do not match exactly.
    Delegate->emitTableEntries(TableInfo);

    // Now that we've emitted the body of the handler, update the NumToSkip
    // of the filter itself to be able to skip forward when false.
    if (PrevFilter)
      Table.patchNumToSkip(PrevFilter, Table.size());
  }

  if (VariableFC) {
    // Each scope should always have at least one filter value to check for.
    assert(PrevFilter != 0 && "empty filter set!");
    TableInfo.popScope();
    PrevFilter = 0; // Don't re-process the filter's fallthrough.

    // Delegate to the sub filter chooser for further decoding.
    VariableFC->emitTableEntries(TableInfo);
  }

  // If there is no fallthrough and the final filter was not in the outermost
  // scope, then it must be fixed up according to the enclosing scope rather
  // than the current position.
  if (PrevFilter)
    TableInfo.FixupStack.back().push_back(PrevFilter);
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
    if (*(I + NumToSkip) == MCD::OPC_Fail)
      OS << " (Fail)";
  };

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
    case MCD::OPC_FilterValue:
    case MCD::OPC_FilterValueOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_FilterValueOrFail;
      OS << "  MCD::OPC_FilterValue" << (IsFail ? "OrFail, " : ", ");
      // The filter value is ULEB128 encoded.
      emitULEB128(I, OS);

      if (!IsFail) {
        uint32_t NumToSkip = emitNumToSkip(I, OS);
        emitNumToSkipComment(NumToSkip);
      }
      OS << '\n';
      break;
    }
    case MCD::OPC_CheckField:
    case MCD::OPC_CheckFieldOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_CheckFieldOrFail;
      OS << "  MCD::OPC_CheckField" << (IsFail ? "OrFail, " : ", ");
      // ULEB128 encoded start value.
      emitULEB128(I, OS);
      // 8-bit length.
      unsigned Len = *I++;
      OS << Len << ", ";
      // ULEB128 encoded field value.
      emitULEB128(I, OS);

      if (!IsFail) {
        uint32_t NumToSkip = emitNumToSkip(I, OS);
        emitNumToSkipComment(NumToSkip);
      }
      OS << '\n';
      break;
    }
    case MCD::OPC_CheckPredicate:
    case MCD::OPC_CheckPredicateOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_CheckPredicateOrFail;

      OS << "  MCD::OPC_CheckPredicate" << (IsFail ? "OrFail, " : ", ");
      emitULEB128(I, OS);

      if (!IsFail) {
        uint32_t NumToSkip = emitNumToSkip(I, OS);
        emitNumToSkipComment(NumToSkip);
      }
      OS << '\n';
      break;
    }
    case MCD::OPC_Decode:
    case MCD::OPC_TryDecode:
    case MCD::OPC_TryDecodeOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_TryDecodeOrFail;
      bool IsTry = DecoderOp == MCD::OPC_TryDecode || IsFail;
      // Decode the Opcode value.
      const char *ErrMsg = nullptr;
      unsigned Opc = decodeULEB128(&*I, nullptr, EndPtr, &ErrMsg);
      assert(ErrMsg == nullptr && "ULEB128 value too large!");

      OS << "  MCD::OPC_" << (IsTry ? "Try" : "") << "Decode"
         << (IsFail ? "OrFail, " : ", ");
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

      // Fallthrough for OPC_TryDecode.
      if (!IsFail) {
        uint32_t NumToSkip = emitNumToSkip(I, OS);
        OS << "// Opcode: " << Encodings[EncodingID].getName()
           << ", DecodeIdx: " << DecodeIdx;
        emitNumToSkipComment(NumToSkip, /*InComment=*/true);
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
    case MCD::OPC_Fail:
      OS << "  MCD::OPC_Fail,\n";
      break;
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
                                         DecoderSet &Decoders) const {
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

  if (UseFnTableInDecodeToMCInst) {
    // Emit a function for each case first.
    for (const auto &[Index, Decoder] : enumerate(Decoders)) {
      OS << "template <typename InsnType>\n";
      OS << "static DecodeStatus decodeFn" << Index << "(" << DecodeParams
         << ") {\n";
      OS << "  using namespace llvm::MCD;\n";
      OS << "  " << TmpTypeDecl;
      OS << "  [[maybe_unused]] TmpType tmp;\n";
      OS << Decoder;
      OS << "  return S;\n";
      OS << "}\n\n";
    }
  }

  OS << "// Handling " << Decoders.size() << " cases.\n";
  OS << "template <typename InsnType>\n";
  OS << "static DecodeStatus decodeToMCInst(unsigned Idx, " << DecodeParams
     << ") {\n";
  OS << "  using namespace llvm::MCD;\n";
  OS << "  DecodeComplete = true;\n";

  if (UseFnTableInDecodeToMCInst) {
    // Build a table of function pointers
    OS << "  using DecodeFnTy = DecodeStatus (*)(" << DecodeParams << ");\n";
    OS << "  static constexpr DecodeFnTy decodeFnTable[] = {\n";
    for (size_t Index : llvm::seq(Decoders.size()))
      OS << "    decodeFn" << Index << ",\n";
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

void FilterChooser::emitBinaryParser(raw_ostream &OS, indent Indent,
                                     const OperandInfo &OpInfo) const {
  const std::string &Decoder = OpInfo.Decoder;

  bool UseInsertBits = OpInfo.numFields() != 1 || OpInfo.InitValue != 0;

  if (UseInsertBits) {
    OS << Indent << "tmp = 0x";
    OS.write_hex(OpInfo.InitValue);
    OS << ";\n";
  }

  for (const EncodingField &EF : OpInfo) {
    OS << Indent;
    if (UseInsertBits)
      OS << "insertBits(tmp, ";
    else
      OS << "tmp = ";
    OS << "fieldFromInstruction(insn, " << EF.Base << ", " << EF.Width << ')';
    if (UseInsertBits)
      OS << ", " << EF.Offset << ", " << EF.Width << ')';
    else if (EF.Offset != 0)
      OS << " << " << EF.Offset;
    OS << ";\n";
  }

  if (!Decoder.empty()) {
    OS << Indent << "if (!Check(S, " << Decoder
       << "(MI, tmp, Address, Decoder))) { "
       << (OpInfo.HasCompleteDecoder ? "" : "DecodeComplete = false; ")
       << "return MCDisassembler::Fail; }\n";
  } else {
    OS << Indent << "MI.addOperand(MCOperand::createImm(tmp));\n";
  }
}

void FilterChooser::emitDecoder(raw_ostream &OS, indent Indent,
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
    if (Op.numFields())
      emitBinaryParser(OS, Indent, Op);
}

unsigned FilterChooser::getDecoderIndex(DecoderSet &Decoders,
                                        unsigned EncodingID) const {
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
  Decoders.insert(CachedHashString(Decoder));
  // Now figure out the index for when we write out the table.
  DecoderSet::const_iterator P = find(Decoders, Decoder.str());
  return std::distance(Decoders.begin(), P);
}

// If ParenIfBinOp is true, print a surrounding () if Val uses && or ||.
bool FilterChooser::emitPredicateMatchAux(const Init &Val, bool ParenIfBinOp,
                                          raw_ostream &OS) const {
  if (const auto *D = dyn_cast<DefInit>(&Val)) {
    if (!D->getDef()->isSubClassOf("SubtargetFeature"))
      return true;
    OS << "Bits[" << Emitter->PredicateNamespace << "::" << D->getAsString()
       << "]";
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

bool FilterChooser::emitPredicateMatch(raw_ostream &OS,
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

bool FilterChooser::doesOpcodeNeedPredicate(unsigned EncodingID) const {
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

unsigned FilterChooser::getPredicateIndex(DecoderTableInfo &TableInfo,
                                          StringRef Predicate) const {
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

void FilterChooser::emitPredicateTableEntry(DecoderTableInfo &TableInfo,
                                            unsigned EncodingID) const {
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
  unsigned PIdx = getPredicateIndex(TableInfo, PS.str());

  const uint8_t DecoderOp = TableInfo.isOutermostScope()
                                ? MCD::OPC_CheckPredicateOrFail
                                : MCD::OPC_CheckPredicate;
  TableInfo.Table.push_back(DecoderOp);
  TableInfo.Table.insertULEB128(PIdx);

  if (DecoderOp == MCD::OPC_CheckPredicate) {
    // Push location for NumToSkip backpatching.
    TableInfo.FixupStack.back().push_back(TableInfo.Table.insertNumToSkip());
  }
}

void FilterChooser::emitSoftFailTableEntry(DecoderTableInfo &TableInfo,
                                           unsigned EncodingID) const {
  const InstructionEncoding &Encoding = Encodings[EncodingID];
  const KnownBits &InstBits = Encoding.getInstBits();
  const APInt &SFBits = Encoding.getSoftFailBits();

  if (SFBits.isZero())
    return;

  unsigned EncodingWidth = InstBits.getBitWidth();
  APInt PositiveMask(EncodingWidth, 0);
  APInt NegativeMask(EncodingWidth, 0);
  for (unsigned i = 0; i != EncodingWidth; ++i) {
    if (!SFBits[i])
      continue;

    if (InstBits.Zero[i]) {
      // The bit is meant to be false, so emit a check to see if it is true.
      PositiveMask.setBit(i);
    } else if (InstBits.One[i]) {
      // The bit is meant to be true, so emit a check to see if it is false.
      NegativeMask.setBit(i);
    }
  }

  bool NeedPositiveMask = PositiveMask.getBoolValue();
  bool NeedNegativeMask = NegativeMask.getBoolValue();

  if (!NeedPositiveMask && !NeedNegativeMask)
    return;

  TableInfo.Table.push_back(MCD::OPC_SoftFail);
  TableInfo.Table.insertULEB128(PositiveMask.getZExtValue());
  TableInfo.Table.insertULEB128(NegativeMask.getZExtValue());
}

// Emits table entries to decode the singleton.
void FilterChooser::emitSingletonTableEntry(DecoderTableInfo &TableInfo,
                                            unsigned EncodingID) const {
  const InstructionEncoding &Encoding = Encodings[EncodingID];
  KnownBits EncodingBits = Encoding.getMandatoryBits();

  // Look for islands of undecoded bits of the singleton.
  std::vector<Island> Islands = getIslands(EncodingBits);

  // Emit the predicate table entry if one is needed.
  emitPredicateTableEntry(TableInfo, EncodingID);

  // Check any additional encoding fields needed.
  for (const Island &Ilnd : reverse(Islands)) {
    assert(isUInt<8>(Ilnd.NumBits) && "NumBits overflowed uint8 table entry!");
    const uint8_t DecoderOp = TableInfo.isOutermostScope()
                                  ? MCD::OPC_CheckFieldOrFail
                                  : MCD::OPC_CheckField;
    TableInfo.Table.push_back(DecoderOp);

    TableInfo.Table.insertULEB128(Ilnd.StartBit);
    TableInfo.Table.push_back(Ilnd.NumBits);
    TableInfo.Table.insertULEB128(Ilnd.FieldVal);

    if (DecoderOp == MCD::OPC_CheckField) {
      // Allocate space in the table for fixup so all our relative position
      // calculations work OK even before we fully resolve the real value here.

      // Push location for NumToSkip backpatching.
      TableInfo.FixupStack.back().push_back(TableInfo.Table.insertNumToSkip());
    }
  }

  // Check for soft failure of the match.
  emitSoftFailTableEntry(TableInfo, EncodingID);

  unsigned DIdx = getDecoderIndex(TableInfo.Decoders, EncodingID);

  // Produce OPC_Decode or OPC_TryDecode opcode based on the information
  // whether the instruction decoder is complete or not. If it is complete
  // then it handles all possible values of remaining variable/unfiltered bits
  // and for any value can determine if the bitpattern is a valid instruction
  // or not. This means OPC_Decode will be the final step in the decoding
  // process. If it is not complete, then the Fail return code from the
  // decoder method indicates that additional processing should be done to see
  // if there is any other instruction that also matches the bitpattern and
  // can decode it.
  const uint8_t DecoderOp =
      Encoding.hasCompleteDecoder()
          ? MCD::OPC_Decode
          : (TableInfo.isOutermostScope() ? MCD::OPC_TryDecodeOrFail
                                          : MCD::OPC_TryDecode);
  TableInfo.Table.push_back(DecoderOp);
  const Record *InstDef = Encodings[EncodingID].getInstruction()->TheDef;
  TableInfo.Table.insertULEB128(Emitter->getTarget().getInstrIntValue(InstDef));
  TableInfo.Table.insertULEB128(DIdx);

  if (DecoderOp == MCD::OPC_TryDecode) {
    // Push location for NumToSkip backpatching.
    TableInfo.FixupStack.back().push_back(TableInfo.Table.insertNumToSkip());
  }
}

// Emits table entries to decode the singleton, and then to decode the rest.
void FilterChooser::emitSingletonTableEntry(DecoderTableInfo &TableInfo) const {
  // complex singletons need predicate checks from the first singleton
  // to refer forward to the variable filterchooser that follows.
  TableInfo.pushScope();
  emitSingletonTableEntry(TableInfo, *SingletonEncodingID);
  TableInfo.popScope();

  VariableFC->emitTableEntries(TableInfo);
}

// reportRegion is a helper function for filterProcessor to mark a region as
// eligible for use as a filter region.
void FilterChooser::reportRegion(std::vector<std::unique_ptr<Filter>> &Filters,
                                 bitAttr_t RA, unsigned StartBit,
                                 unsigned BitIndex, bool AllowMixed) const {
  if (AllowMixed ? RA == ATTR_MIXED : RA == ATTR_ALL_SET)
    Filters.push_back(std::make_unique<Filter>(Encodings, EncodingIDs, StartBit,
                                               BitIndex - StartBit));
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
        reportRegion(Filters, RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        break;
      case ATTR_ALL_UNSET:
        reportRegion(Filters, RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        reportRegion(Filters, RA, StartBit, BitIndex, AllowMixed);
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
        reportRegion(Filters, RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        reportRegion(Filters, RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
        reportRegion(Filters, RA, StartBit, BitIndex, AllowMixed);
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
    reportRegion(Filters, RA, StartBit, FilterWidth, AllowMixed);
    break;
  case ATTR_ALL_UNSET:
    break;
  case ATTR_MIXED:
    reportRegion(Filters, RA, StartBit, FilterWidth, AllowMixed);
    break;
  }

  // We have finished with the filter processings.  Now it's time to choose
  // the best performing filter.
  unsigned BestIndex = 0;
  bool AllUseless = true;
  unsigned BestScore = 0;

  for (const auto &[Idx, Filter] : enumerate(Filters)) {
    unsigned Usefulness = Filter->usefulness();

    if (Usefulness)
      AllUseless = false;

    if (Usefulness > BestScore) {
      BestIndex = Idx;
      BestScore = Usefulness;
    }
  }

  if (AllUseless)
    return nullptr;

  return std::move(Filters[BestIndex]);
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
  if (EncodingIDs.size() < 2)
    return;

  std::unique_ptr<Filter> BestFilter = findBestFilter();
  if (BestFilter) {
    applyFilter(*BestFilter);
    return;
  }

  // Print out useful conflict information for postmortem analysis.
  errs() << "Decoding Conflict:\n";
  dump();
  PrintFatalError("Decoding conflict encountered");
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

// emitTableEntries - Emit state machine entries to decode our share of
// instructions.
void FilterChooser::emitTableEntries(DecoderTableInfo &TableInfo) const {
  if (EncodingIDs.size() == 1) {
    // There is only one instruction in the set, which is great!
    // Call emitSingletonDecoder() to see whether there are any remaining
    // encodings bits.
    emitSingletonTableEntry(TableInfo, EncodingIDs[0]);
    return;
  }

  // Use the best filter to do the decoding!
  if (SingletonEncodingID)
    emitSingletonTableEntry(TableInfo);
  else
    emitTableEntry(TableInfo);
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
  SoftFailBits = APInt(VLI.size(), 0);

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

void InstructionEncoding::parseFixedLenEncoding(const BitsInit &Bits) {
  InstBits = KnownBits(Bits.getNumBits());
  SoftFailBits = APInt(Bits.getNumBits(), 0);

  // Parse Inst field.
  for (auto [I, V] : enumerate(Bits.getBits())) {
    if (const auto *B = dyn_cast<BitInit>(V)) {
      if (B->getValue())
        InstBits.One.setBit(I);
      else
        InstBits.Zero.setBit(I);
    }
  }

  // Parse SoftFail field.
  if (const RecordVal *SoftFailField = EncodingDef->getValue("SoftFail")) {
    const auto *SFBits = dyn_cast<BitsInit>(SoftFailField->getValue());
    if (!SFBits || SFBits->getNumBits() != Bits.getNumBits()) {
      PrintNote(EncodingDef->getLoc(), "in record");
      PrintFatalError(SoftFailField,
                      formatv("SoftFail field, if defined, must be "
                              "of the same type as Inst, which is bits<{}>",
                              Bits.getNumBits()));
    }
    for (auto [I, V] : enumerate(SFBits->getBits())) {
      if (const auto *B = dyn_cast<BitInit>(V); B && B->getValue()) {
        if (!InstBits.Zero[I] && !InstBits.One[I]) {
          PrintNote(EncodingDef->getLoc(), "in record");
          PrintError(SoftFailField,
                     formatv("SoftFail{{{0}} = 1 requires Inst{{{0}} "
                             "to be fully defined (0 or 1, not '?')",
                             I));
        }
        SoftFailBits.setBit(I);
      }
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
static void addOneOperandFields(const Record *EncodingDef, const BitsInit &Bits,
                                std::map<StringRef, StringRef> &TiedNames,
                                StringRef OpName, OperandInfo &OpInfo) {
  // Some bits of the operand may be required to be 1 depending on the
  // instruction's encoding. Collect those bits.
  if (const RecordVal *EncodedValue = EncodingDef->getValue(OpName))
    if (const BitsInit *OpBits = dyn_cast<BitsInit>(EncodedValue->getValue()))
      for (unsigned I = 0; I < OpBits->getNumBits(); ++I)
        if (const BitInit *OpBit = dyn_cast<BitInit>(OpBits->getBit(I)))
          if (OpBit->getValue())
            OpInfo.InitValue |= 1ULL << I;

  for (unsigned I = 0, J = 0; I != Bits.getNumBits(); I = J) {
    const VarInit *Var;
    unsigned Offset = 0;
    for (; J != Bits.getNumBits(); ++J) {
      const VarBitInit *BJ = dyn_cast<VarBitInit>(Bits.getBit(J));
      if (BJ) {
        Var = dyn_cast<VarInit>(BJ->getBitVar());
        if (I == J)
          Offset = BJ->getBitNum();
        else if (BJ->getBitNum() != Offset + J - I)
          break;
      } else {
        Var = dyn_cast<VarInit>(Bits.getBit(J));
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
  const Record &Def = *Inst->TheDef;

  // Gather the outputs/inputs of the instruction, so we can find their
  // positions in the encoding.  This assumes for now that they appear in the
  // MCInst in the order that they're listed.
  std::vector<std::pair<const Init *, StringRef>> InOutOperands;
  const DagInit *Out = Def.getValueAsDag("OutOperandList");
  const DagInit *In = Def.getValueAsDag("InOperandList");
  for (const auto &[Idx, Arg] : enumerate(Out->getArgs()))
    InOutOperands.emplace_back(Arg, Out->getArgNameStr(Idx));
  for (const auto &[Idx, Arg] : enumerate(In->getArgs()))
    InOutOperands.emplace_back(Arg, In->getArgNameStr(Idx));

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
  for (const auto &Op : InOutOperands) {
    const Init *OpInit = Op.first;
    StringRef OpName = Op.second;

    // We're ready to find the instruction encoding locations for this
    // operand.

    // First, find the operand type ("OpInit"), and sub-op names
    // ("SubArgDag") if present.
    const DagInit *SubArgDag = dyn_cast<DagInit>(OpInit);
    if (SubArgDag)
      OpInit = SubArgDag->getOperator();
    const Record *OpTypeRec = cast<DefInit>(OpInit)->getDef();
    // Lookup the sub-operands from the operand type record (note that only
    // Operand subclasses have MIOperandInfo, see CodeGenInstruction.cpp).
    const DagInit *SubOps = OpTypeRec->isSubClassOf("Operand")
                                ? OpTypeRec->getValueAsDag("MIOperandInfo")
                                : nullptr;

    // Lookup the decoder method and construct a new OperandInfo to hold our
    // result.
    OperandInfo OpInfo = getOpInfo(OpTypeRec);

    // If we have named sub-operands...
    if (SubArgDag) {
      // Then there should not be a custom decoder specified on the top-level
      // type.
      if (!OpInfo.Decoder.empty()) {
        PrintError(EncodingDef,
                   "DecoderEmitter: operand \"" + OpName + "\" has type \"" +
                       OpInit->getAsString() +
                       "\" with a custom DecoderMethod, but also named "
                       "sub-operands.");
        continue;
      }

      // Decode each of the sub-ops separately.
      assert(SubOps && SubArgDag->getNumArgs() == SubOps->getNumArgs());
      for (const auto &[I, Arg] : enumerate(SubOps->getArgs())) {
        StringRef SubOpName = SubArgDag->getArgNameStr(I);
        OperandInfo SubOpInfo = getOpInfo(cast<DefInit>(Arg)->getDef());

        addOneOperandFields(EncodingDef, Bits, TiedNames, SubOpName, SubOpInfo);
        Operands.push_back(std::move(SubOpInfo));
      }
      continue;
    }

    // Otherwise, if we have an operand with sub-operands, but they aren't
    // named...
    if (SubOps && OpInfo.Decoder.empty()) {
      // If it's a single sub-operand, and no custom decoder, use the decoder
      // from the one sub-operand.
      if (SubOps->getNumArgs() == 1)
        OpInfo = getOpInfo(cast<DefInit>(SubOps->getArg(0))->getDef());

      // If we have multiple sub-ops, there'd better have a custom
      // decoder. (Otherwise we don't know how to populate them properly...)
      if (SubOps->getNumArgs() > 1) {
        PrintError(EncodingDef,
                   "DecoderEmitter: operand \"" + OpName +
                       "\" uses MIOperandInfo with multiple ops, but doesn't "
                       "have a custom decoder!");
        debugDumpRecord(*EncodingDef);
        continue;
      }
    }

    addOneOperandFields(EncodingDef, Bits, TiedNames, OpName, OpInfo);
    // FIXME: it should be an error not to find a definition for a given
    // operand, rather than just failing to add it to the resulting
    // instruction! (This is a longstanding bug, which will be addressed in an
    // upcoming change.)
    if (OpInfo.numFields() > 0)
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
  const bool HasTryDecode = OpcodeMask & ((1 << MCD::OPC_TryDecode) |
                                          (1 << MCD::OPC_TryDecodeOrFail));
  const bool HasCheckPredicate =
      OpcodeMask &
      ((1 << MCD::OPC_CheckPredicate) | (1 << MCD::OPC_CheckPredicateOrFail));
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

  OS << R"(
  const uint8_t *Ptr = DecodeTable;
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
    case MCD::OPC_FilterValue:
    case MCD::OPC_FilterValueOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_FilterValueOrFail;
      // Decode the field value.
      uint64_t Val = decodeULEB128AndIncUnsafe(Ptr);
      bool Failed = Val != CurFieldValue;
      unsigned NumToSkip = IsFail ? 0 : decodeNumToSkip(Ptr);

      // Note: Print NumToSkip even for OPC_FilterValueOrFail to simplify debug
      // prints.
      LLVM_DEBUG({
        StringRef OpName = IsFail ? "OPC_FilterValueOrFail" : "OPC_FilterValue";
        dbgs() << Loc << ": " << OpName << '(' << Val << ", " << NumToSkip
                << ") " << (Failed ? "FAIL:" : "PASS:")
                << " continuing at " << (Ptr - DecodeTable) << '\n';
      });

      // Perform the filter operation.
      if (Failed) {
        if (IsFail)
          return MCDisassembler::Fail;
        Ptr += NumToSkip;
      }
      break;
    }
    case MCD::OPC_CheckField:
    case MCD::OPC_CheckFieldOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_CheckFieldOrFail;
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
      unsigned NumToSkip = IsFail ? 0 : decodeNumToSkip(Ptr);

      LLVM_DEBUG({
        StringRef OpName = IsFail ? "OPC_CheckFieldOrFail" : "OPC_CheckField";
        dbgs() << Loc << ": " << OpName << '(' << Start << ", " << Len << ", "
                << ExpectedValue << ", " << NumToSkip << "): FieldValue = "
                << FieldValue << ", ExpectedValue = " << ExpectedValue << ": "
                << (Failed ? "FAIL\n" : "PASS\n");
      });

      // If the actual and expected values don't match, skip or fail.
      if (Failed) {
        if (IsFail)
          return MCDisassembler::Fail;
        Ptr += NumToSkip;
      }
      break;
    })";
  if (HasCheckPredicate) {
    OS << R"(
    case MCD::OPC_CheckPredicate:
    case MCD::OPC_CheckPredicateOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_CheckPredicateOrFail;
      // Decode the Predicate Index value.
      unsigned PIdx = decodeULEB128AndIncUnsafe(Ptr);
      unsigned NumToSkip = IsFail ? 0 : decodeNumToSkip(Ptr);
      // Check the predicate.
      bool Failed = !checkDecoderPredicate(PIdx, Bits);

      LLVM_DEBUG({
        StringRef OpName = IsFail ? "OPC_CheckPredicateOrFail" : "OPC_CheckPredicate";
        dbgs() << Loc << ": " << OpName << '(' << PIdx << ", " << NumToSkip
               << "): " << (Failed ? "FAIL\n" : "PASS\n");
      });

      if (Failed) {
        if (IsFail)
          return MCDisassembler::Fail;
        Ptr += NumToSkip;
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
    case MCD::OPC_TryDecode:
    case MCD::OPC_TryDecodeOrFail: {
      bool IsFail = DecoderOp == MCD::OPC_TryDecodeOrFail;
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128AndIncUnsafe(Ptr);
      unsigned DecodeIdx = decodeULEB128AndIncUnsafe(Ptr);
      unsigned NumToSkip = IsFail ? 0 : decodeNumToSkip(Ptr);

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
      if (IsFail) {
        LLVM_DEBUG(dbgs() << "FAIL: returning FAIL\n");
        return MCDisassembler::Fail;
      }
      // If the decoding was incomplete, skip.
      Ptr += NumToSkip;
      LLVM_DEBUG(dbgs() << "FAIL: continuing at " << (Ptr - DecodeTable) << "\n");
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
    case MCD::OPC_Fail: {
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Fail\n");
      return MCDisassembler::Fail;
    }
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
        std::string DecoderNamespace =
            EncodingDef->getValueAsString("DecoderNamespace").str();
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
    const Record *InstDef = Encodings[EncodingID].getInstruction()->TheDef;
    std::string DecoderNamespace =
        InstDef->getValueAsString("DecoderNamespace").str();
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

DecoderEmitter::DecoderEmitter(const RecordKeeper &RK,
                               StringRef PredicateNamespace)
    : RK(RK), Target(RK), CGH(Target.getHwModes()),
      PredicateNamespace(PredicateNamespace) {
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
)";

  // Do extra bookkeeping for variable-length encodings.
  std::vector<unsigned> InstrLen;
  bool IsVarLenInst = Target.hasVariableLengthEncodings();
  unsigned MaxInstLen = 0;
  if (IsVarLenInst) {
    InstrLen.resize(Target.getInstructions().size(), 0);
    for (const InstructionEncoding &Encoding : Encodings) {
      MaxInstLen = std::max(MaxInstLen, Encoding.getBitWidth());
      InstrLen[Target.getInstrIntValue(Encoding.getInstruction()->TheDef)] =
          Encoding.getBitWidth();
    }
  }

  // Map of (namespace, hwmode, size) tuple to encoding IDs.
  std::map<std::tuple<StringRef, unsigned, unsigned>, std::vector<unsigned>>
      EncMap;
  for (const auto &[HwModeID, EncodingIDs] : EncodingIDsByHwMode) {
    for (unsigned EncodingID : EncodingIDs) {
      const InstructionEncoding &Encoding = Encodings[EncodingID];
      const Record *EncodingDef = Encoding.getRecord();
      unsigned Size = EncodingDef->getValueAsInt("Size");
      StringRef DecoderNamespace =
          EncodingDef->getValueAsString("DecoderNamespace");
      EncMap[{DecoderNamespace, HwModeID, Size}].push_back(EncodingID);
    }
  }

  DecoderTableInfo TableInfo;
  unsigned OpcodeMask = 0;

  for (const auto &[Key, EncodingIDs] : EncMap) {
    auto [DecoderNamespace, HwModeID, Size] = Key;
    const unsigned BitWidth = IsVarLenInst ? MaxInstLen : 8 * Size;
    // Emit the decoder for this (namespace, hwmode, width) combination.
    FilterChooser FC(Encodings, EncodingIDs, BitWidth, this);

    // The decode table is cleared for each top level decoder function. The
    // predicates and decoders themselves, however, are shared across all
    // decoders to give more opportunities for uniqueing.
    TableInfo.Table.clear();
    TableInfo.pushScope();
    FC.emitTableEntries(TableInfo);
    // Any NumToSkip fixups in the top level scope can resolve to the
    // OPC_Fail at the end of the table.
    assert(TableInfo.isOutermostScope() && "fixup stack phasing error!");
    TableInfo.popScope();

    TableInfo.Table.push_back(MCD::OPC_Fail);

    // Print the table to the output stream.
    OpcodeMask |= emitTable(OS, TableInfo.Table, DecoderNamespace, HwModeID,
                            BitWidth, EncodingIDs);
  }

  // For variable instruction, we emit a instruction length table
  // to let the decoder know how long the instructions are.
  // You can see example usage in M68k's disassembler.
  if (IsVarLenInst)
    emitInstrLenTable(OS, InstrLen);

  const bool HasCheckPredicate =
      OpcodeMask &
      ((1 << MCD::OPC_CheckPredicate) | (1 << MCD::OPC_CheckPredicateOrFail));

  // Emit the predicate function.
  if (HasCheckPredicate)
    emitPredicateFunction(OS, TableInfo.Predicates);

  // Emit the decoder function.
  emitDecoderFunction(OS, TableInfo.Decoders);

  // Emit the main entry point for the decoder, decodeInstruction().
  emitDecodeInstruction(OS, IsVarLenInst, OpcodeMask);

  OS << "\n} // namespace\n";
}

void llvm::EmitDecoder(const RecordKeeper &RK, raw_ostream &OS,
                       StringRef PredicateNamespace) {
  DecoderEmitter(RK, PredicateNamespace).run(OS);
}
