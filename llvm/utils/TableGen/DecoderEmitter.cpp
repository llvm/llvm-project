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
#include "Common/InstructionEncoding.h"
#include "Common/SubtargetFeatureInfo.h"
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

using PredicateSet = SetVector<CachedHashString>;
using DecoderSet = SetVector<CachedHashString>;

struct DecoderTableInfo {
  PredicateSet Predicates;
  DecoderSet Decoders;
  bool HasCheckPredicate;
  bool HasSoftFail;

  void insertPredicate(StringRef Predicate) {
    Predicates.insert(CachedHashString(Predicate));
  }

  void insertDecoder(StringRef Decoder) {
    Decoders.insert(CachedHashString(Decoder));
  }

  unsigned getPredicateIndex(StringRef Predicate) const {
    auto I = find(Predicates, Predicate);
    assert(I != Predicates.end());
    return std::distance(Predicates.begin(), I);
  }

  unsigned getDecoderIndex(StringRef Decoder) const {
    auto I = find(Decoders, Decoder);
    assert(I != Decoders.end());
    return std::distance(Decoders.begin(), I);
  }
};

using NamespacesHwModesMap = std::map<StringRef, std::set<unsigned>>;

class DecoderTreeNode;

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
  void emitTable(formatted_raw_ostream &OS, DecoderTableInfo &TableInfo,
                 StringRef Namespace, unsigned HwModeID, unsigned BitWidth,
                 const DecoderTreeNode *Tree) const;
  void emitInstrLenTable(formatted_raw_ostream &OS,
                         ArrayRef<unsigned> InstrLen) const;
  void emitPredicateFunction(formatted_raw_ostream &OS,
                             const PredicateSet &Predicates) const;
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

struct EncodingIsland {
  unsigned StartBit;
  unsigned NumBits;
  uint64_t FieldVal;
};

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
  friend class DecoderTreeBuilder;

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

void DecoderEmitter::emitInstrLenTable(formatted_raw_ostream &OS,
                                       ArrayRef<unsigned> InstrLen) const {
  OS << "static const uint8_t InstrLenTable[] = {\n";
  for (unsigned Len : InstrLen)
    OS << Len << ",\n";
  OS << "};\n\n";
}

void DecoderEmitter::emitPredicateFunction(
    formatted_raw_ostream &OS, const PredicateSet &Predicates) const {
  // The predicate function is just a big switch statement based on the
  // input predicate index.
  OS << "static bool checkDecoderPredicate(unsigned Idx, const FeatureBitset "
        "&FB) {\n";
  OS << "  switch (Idx) {\n";
  OS << "  default: llvm_unreachable(\"Invalid index!\");\n";
  for (const auto &[Index, Predicate] : enumerate(Predicates)) {
    OS << "  case " << Index << ":\n";
    OS << "    return " << Predicate << ";\n";
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
static std::vector<EncodingIsland> getIslands(const KnownBits &EncodingBits,
                                              const KnownBits &FilterBits) {
  std::vector<EncodingIsland> Islands;
  uint64_t FieldVal;
  unsigned StartBit;

  bool OnIsland = false;
  unsigned FilterWidth = FilterBits.getBitWidth();
  for (unsigned I = 0; I != FilterWidth; ++I) {
    bool IsKnown = EncodingBits.Zero[I] || EncodingBits.One[I];
    bool IsFiltered = FilterBits.Zero[I] || FilterBits.One[I];
    if (!IsFiltered && IsKnown) {
      if (OnIsland) {
        // Accumulate island bits.
        FieldVal |= static_cast<uint64_t>(EncodingBits.One[I])
                    << (I - StartBit);
      } else {
        // Onto an island.
        StartBit = I;
        FieldVal = static_cast<uint64_t>(EncodingBits.One[I]);
        OnIsland = true;
      }
    } else if (OnIsland) {
      // Into the water.
      Islands.push_back({StartBit, I - StartBit, FieldVal});
      OnIsland = false;
    }
  }

  if (OnIsland)
    Islands.push_back({StartBit, FilterWidth - StartBit, FieldVal});

  return Islands;
}

static void emitBinaryParser(raw_ostream &OS, indent Indent,
                             const InstructionEncoding &Encoding,
                             const OperandInfo &OpInfo) {
  if (OpInfo.HasNoEncoding) {
    // If an operand has no encoding, the old behavior is to not decode it
    // automatically and let the target do it. This is error-prone, so the
    // new behavior is to report an error.
    if (!IgnoreNonDecodableOperands)
      PrintError(Encoding.getRecord()->getLoc(),
                 "could not find field for operand '" + OpInfo.Name + "'");
    return;
  }

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

  if (OpInfo.fields().empty()) {
    // Only a constant part. The old behavior is to not decode this operand.
    if (IgnoreFullyDefinedOperands)
      return;
    // Initialize `tmp` with the constant part.
    OS << Indent << "tmp = " << format_hex(*OpInfo.InitValue, 0) << ";\n";
  } else if (OpInfo.fields().size() == 1 && !OpInfo.InitValue.value_or(0)) {
    // One variable part and no/zero constant part. Initialize `tmp` with the
    // variable part.
    auto [Base, Width, Offset] = OpInfo.fields().front();
    OS << Indent << "tmp = fieldFromInstruction(insn, " << Base << ", " << Width
       << ')';
    if (Offset)
      OS << " << " << Offset;
    OS << ";\n";
  } else {
    // General case. Initialize `tmp` with the constant part, if any, and
    // insert the variable parts into it.
    OS << Indent << "tmp = " << format_hex(OpInfo.InitValue.value_or(0), 0)
       << ";\n";
    for (auto [Base, Width, Offset] : OpInfo.fields()) {
      OS << Indent << "tmp |= fieldFromInstruction(insn, " << Base << ", "
         << Width << ')';
      if (Offset)
        OS << " << " << Offset;
      OS << ";\n";
    }
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

static std::string getDecoderString(const InstructionEncoding &Encoding) {
  std::string Decoder;
  raw_string_ostream OS(Decoder);
  indent Indent(UseFnTableInDecodeToMCInst ? 2 : 4);

  // If a custom instruction decoder was specified, use that.
  StringRef DecoderMethod = Encoding.getDecoderMethod();
  if (!DecoderMethod.empty()) {
    OS << Indent << "if (!Check(S, " << DecoderMethod
       << "(MI, insn, Address, Decoder))) { "
       << (Encoding.hasCompleteDecoder() ? "" : "DecodeComplete = false; ")
       << "return MCDisassembler::Fail; }\n";
  } else {
    for (const OperandInfo &Op : Encoding.getOperands())
      emitBinaryParser(OS, Indent, Encoding, Op);
  }
  return Decoder;
}

static std::string getPredicateString(const InstructionEncoding &Encoding,
                                      StringRef TargetName) {
  std::vector<const Record *> Predicates =
      Encoding.getRecord()->getValueAsListOfDefs("Predicates");
  auto It = llvm::find_if(Predicates, [](const Record *R) {
    return R->getValueAsBit("AssemblerMatcherPredicate");
  });
  if (It == Predicates.end())
    return std::string();

  std::string Predicate;
  raw_string_ostream OS(Predicate);
  SubtargetFeatureInfo::emitMCPredicateCheck(OS, TargetName, Predicates);
  return Predicate;
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
      std::vector<EncodingIsland> Islands =
          getIslands(EncodingBits, FilterBits);
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

// emitDecodeInstruction - Emit the templated helper function
// decodeInstruction().
static void emitDecodeInstruction(formatted_raw_ostream &OS, bool IsVarLenInst,
                                  const DecoderTableInfo &TableInfo) {
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
  if (TableInfo.HasCheckPredicate)
    OS << "  const FeatureBitset &Bits = STI.getFeatureBits();\n";
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
    case OPC_Scope: {
      unsigned NumToSkip = decodeULEB128AndIncUnsafe(Ptr);
      const uint8_t *SkipTo = Ptr + NumToSkip;
      ScopeStack.push_back(SkipTo);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Scope(" << SkipTo - DecodeTable
                        << ")\n");
      continue;
    }
    case OPC_SwitchField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(Ptr);
      unsigned Len = *Ptr++;)";
  if (IsVarLenInst)
    OS << "\n      makeUp(insn, Start + Len);";
  OS << R"(
      uint64_t FieldValue = fieldFromInstruction(insn, Start, Len);
      uint64_t CaseValue;
      unsigned CaseSize;
      while (true) {
        CaseValue = decodeULEB128AndIncUnsafe(Ptr);
        CaseSize = decodeULEB128AndIncUnsafe(Ptr);
        if (FieldValue == CaseValue || !CaseSize)
          break;
        Ptr += CaseSize;
      }
      if (FieldValue == CaseValue) {
        LLVM_DEBUG(dbgs() << Loc << ": OPC_SwitchField(" << Start << ", " << Len
                          << "): " << FieldValue << '\n');
        continue;
      }
      break;
    }
    case OPC_CheckField: {
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
      if (!Failed)
        continue;
      break;
    })";
  if (TableInfo.HasCheckPredicate) {
    OS << R"(
    case OPC_CheckPredicate: {
      // Decode the Predicate Index value.
      unsigned PIdx = decodeULEB128AndIncUnsafe(Ptr);
      // Check the predicate.
      bool Failed = !checkDecoderPredicate(PIdx, Bits);

      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckPredicate(" << PIdx << "): "
                        << (Failed ? "FAIL, " : "PASS\n"););
      if (!Failed)
        continue;
      break;
    })";
  }
  OS << R"(
    case OPC_Decode: {
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
      S = decodeToMCInst(DecodeIdx, S, insn, MI, Address, DisAsm,
                         DecodeComplete);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Decode: opcode " << Opc
                        << ", using decoder " << DecodeIdx << ": "
                        << (S ? "PASS, " : "FAIL, "));

      if (DecodeComplete) {
        LLVM_DEBUG(dbgs() << "decoding complete\n");
        return S;
      }
      assert(S == MCDisassembler::Fail);
      // Reset decode status. This also drops a SoftFail status that could be
      // set before the decode attempt.
      S = MCDisassembler::Success;
      break;
    })";
  if (TableInfo.HasSoftFail) {
    OS << R"(
    case OPC_SoftFail: {
      // Decode the mask values.
      uint64_t PositiveMask = decodeULEB128AndIncUnsafe(Ptr);
      uint64_t NegativeMask = decodeULEB128AndIncUnsafe(Ptr);
      bool Failed = (insn & PositiveMask) != 0 || (~insn & NegativeMask) != 0;
      if (Failed)
        S = MCDisassembler::SoftFail;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_SoftFail: " << (Failed ? "FAIL\n" : "PASS\n"));
      continue;
    })";
  }
  OS << R"(
    }
    if (ScopeStack.empty()) {
      LLVM_DEBUG(dbgs() << "returning Fail\n");
      return MCDisassembler::Fail;
    }
    Ptr = ScopeStack.pop_back_val();
    LLVM_DEBUG(dbgs() << "continuing at " << Ptr - DecodeTable << '\n');
  }
  llvm_unreachable("bogosity detected in disassembler state machine!");
}

)";
}

namespace {

class DecoderTreeNode {
public:
  virtual ~DecoderTreeNode() = default;

  enum KindTy {
    CheckAny,
    CheckAll,
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

class CheckManyNode : public DecoderTreeNode {
  SmallVector<std::unique_ptr<DecoderTreeNode>, 0> Children;

  static const DecoderTreeNode *
  mapElement(decltype(Children)::const_reference Element) {
    return Element.get();
  }

protected:
  explicit CheckManyNode(KindTy Kind) : DecoderTreeNode(Kind) {}

public:
  void addChild(std::unique_ptr<DecoderTreeNode> Child) {
    Children.push_back(std::move(Child));
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

class CheckAnyNode : public CheckManyNode {
public:
  CheckAnyNode() : CheckManyNode(CheckAny) {}
};

class CheckAllNode : public CheckManyNode {
public:
  CheckAllNode() : CheckManyNode(CheckAll) {}
};

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

class CheckPredicateNode : public DecoderTreeNode {
  std::string PredicateString;

public:
  explicit CheckPredicateNode(std::string PredicateString)
      : DecoderTreeNode(CheckPredicate),
        PredicateString(std::move(PredicateString)) {}

  StringRef getPredicateString() const { return PredicateString; }
};

class SoftFailNode : public DecoderTreeNode {
  uint64_t PositiveMask, NegativeMask;

public:
  SoftFailNode(uint64_t PositiveMask, uint64_t NegativeMask)
      : DecoderTreeNode(SoftFail), PositiveMask(PositiveMask),
        NegativeMask(NegativeMask) {}

  uint64_t getPositiveMask() const { return PositiveMask; }
  uint64_t getNegativeMask() const { return NegativeMask; }
};

class DecodeNode : public DecoderTreeNode {
  const InstructionEncoding &Encoding;
  std::string DecoderString;

public:
  DecodeNode(const InstructionEncoding &Encoding, std::string DecoderString)
      : DecoderTreeNode(Decode), Encoding(Encoding),
        DecoderString(std::move(DecoderString)) {}

  const InstructionEncoding &getEncoding() const { return Encoding; }

  StringRef getDecoderString() const { return DecoderString; }
};

class DecoderTreeBuilder {
  const CodeGenTarget &Target;
  ArrayRef<InstructionEncoding> Encodings;

public:
  DecoderTreeBuilder(const CodeGenTarget &Target,
                     ArrayRef<InstructionEncoding> Encodings)
      : Target(Target), Encodings(Encodings) {}

  std::unique_ptr<DecoderTreeNode> buildTree(const FilterChooser &FC) {
    return convertFilterChooser(&FC);
  }

private:
  std::unique_ptr<DecoderTreeNode>
  convertSingleton(unsigned EncodingID, const KnownBits &FilterBits);

  std::unique_ptr<DecoderTreeNode> convertFilterChooserMap(
      unsigned StartBit, unsigned NumBits,
      const std::map<uint64_t, std::unique_ptr<const FilterChooser>> &FCMap);

  std::unique_ptr<DecoderTreeNode>
  convertFilterChooser(const FilterChooser *FC);
};

class DecoderTableEmitter {
  DecoderTableInfo &TableInfo;
  formatted_raw_ostream OS;
  unsigned IndexWidth;
  unsigned CurrentIndex;
  unsigned CommentIndex;

public:
  DecoderTableEmitter(DecoderTableInfo &TableInfo, raw_ostream &OS)
      : TableInfo(TableInfo), OS(OS) {}

  void emitTable(StringRef TableName, unsigned BitWidth,
                 const DecoderTreeNode *Root);

private:
  void analyzeNode(const DecoderTreeNode *Node) const;

  unsigned computeNodeSize(const DecoderTreeNode *Node) const;
  unsigned computeTableSize(const DecoderTreeNode *Root,
                            unsigned BitWidth) const;

  void emitStartLine();
  void emitOpcode(StringRef Name);
  void emitByte(uint8_t Val);
  void emitUInt8(unsigned Val);
  void emitULEB128(uint64_t Val);
  formatted_raw_ostream &emitComment(indent Indent);

  void emitCheckAnyNode(const CheckAnyNode *N, indent Indent);
  void emitCheckAllNode(const CheckAllNode *N, indent Indent);
  void emitSwitchFieldNode(const SwitchFieldNode *N, indent Indent);
  void emitCheckFieldNode(const CheckFieldNode *N, indent Indent);
  void emitCheckPredicateNode(const CheckPredicateNode *N, indent Indent);
  void emitSoftFailNode(const SoftFailNode *N, indent Indent);
  void emitDecodeNode(const DecodeNode *N, indent Indent);
  void emitNode(const DecoderTreeNode *N, indent Indent);
};

} // namespace

std::unique_ptr<DecoderTreeNode>
DecoderTreeBuilder::convertSingleton(unsigned EncodingID,
                                     const KnownBits &FilterBits) {
  const InstructionEncoding &Encoding = Encodings[EncodingID];
  auto N = std::make_unique<CheckAllNode>();

  std::string Predicate = getPredicateString(Encoding, Target.getName());
  if (!Predicate.empty())
    N->addChild(std::make_unique<CheckPredicateNode>(std::move(Predicate)));

  std::vector<EncodingIsland> Islands =
      getIslands(Encoding.getMandatoryBits(), FilterBits);
  for (const EncodingIsland &Island : reverse(Islands)) {
    N->addChild(std::make_unique<CheckFieldNode>(
        Island.StartBit, Island.NumBits, Island.FieldVal));
  }

  const KnownBits &InstBits = Encoding.getInstBits();
  const APInt &SoftFailMask = Encoding.getSoftFailMask();
  if (!SoftFailMask.isZero()) {
    APInt PositiveMask = InstBits.Zero & SoftFailMask;
    APInt NegativeMask = InstBits.One & SoftFailMask;
    N->addChild(std::make_unique<SoftFailNode>(PositiveMask.getZExtValue(),
                                               NegativeMask.getZExtValue()));
  }

  std::string DecoderString = getDecoderString(Encoding);
  N->addChild(std::make_unique<DecodeNode>(Encoding, std::move(DecoderString)));

  return N;
}

std::unique_ptr<DecoderTreeNode> DecoderTreeBuilder::convertFilterChooserMap(
    unsigned StartBit, unsigned NumBits,
    const std::map<uint64_t, std::unique_ptr<const FilterChooser>> &FCMap) {
  if (FCMap.size() == 1) {
    const auto &[FieldVal, ChildFC] = *FCMap.begin();
    auto N = std::make_unique<CheckAllNode>();
    N->addChild(std::make_unique<CheckFieldNode>(StartBit, NumBits, FieldVal));
    N->addChild(convertFilterChooser(ChildFC.get()));
    return N;
  }
  auto N = std::make_unique<SwitchFieldNode>(StartBit, NumBits);
  for (const auto &[FieldVal, ChildFC] : FCMap)
    N->addCase(FieldVal, convertFilterChooser(ChildFC.get()));
  return N;
}

std::unique_ptr<DecoderTreeNode>
DecoderTreeBuilder::convertFilterChooser(const FilterChooser *FC) {
  auto N = std::make_unique<CheckAnyNode>();

  do {
    if (FC->SingletonEncodingID)
      N->addChild(convertSingleton(*FC->SingletonEncodingID, FC->FilterBits));
    else
      N->addChild(convertFilterChooserMap(FC->StartBit, FC->NumBits,
                                          FC->FilterChooserMap));
    FC = FC->VariableFC.get();
  } while (FC);

  return N;
}

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
  if (SpecializeDecodersPerBitwidth)
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

  // TODO: Emit single OPC_Scope and check at runtime where the scope ends.
  //   Not that it would save much space: OPC_Scope means we've resolved
  //   a conflict, and there aren't many of them. AMDGPU is an exception,
  //   it has ~3.5K OPC_Scope opcodes, but most, if not all scopes contain
  //   only two children.
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

// Emit the decoder state machine table.
void DecoderEmitter::emitTable(formatted_raw_ostream &OS,
                               DecoderTableInfo &TableInfo, StringRef Namespace,
                               unsigned HwModeID, unsigned BitWidth,
                               const DecoderTreeNode *Tree) const {
  SmallString<32> TableName("DecoderTable");
  TableName.append(Namespace);
  if (HwModeID != DefaultMode)
    TableName.append({"_", Target.getHwModes().getModeName(HwModeID)});
  TableName.append(utostr(BitWidth));

  DecoderTableEmitter TableEmitter(TableInfo, OS);
  TableEmitter.emitTable(TableName, BitWidth, Tree);
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
  DecoderTableInfo TableInfo{};

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

      DecoderTreeBuilder TreeBuilder(Target, Encodings);
      std::unique_ptr<DecoderTreeNode> Tree = TreeBuilder.buildTree(FC);

      // The decode table is cleared for each top level decoder function. The
      // predicates and decoders themselves, however, are shared across
      // different decoders to give more opportunities for uniqueing.
      //  - If `SpecializeDecodersPerBitwidth` is enabled, decoders are shared
      //    across all decoder tables for a given bitwidth, else they are shared
      //    across all decoder tables.
      //  - predicates are shared across all decoder tables.
      // Print the table to the output stream.
      emitTable(OS, TableInfo, DecoderNamespace, HwModeID, BitWidth,
                Tree.get());
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

  // Emit the predicate function.
  if (TableInfo.HasCheckPredicate)
    emitPredicateFunction(OS, TableInfo.Predicates);

  // Emit the main entry point for the decoder, decodeInstruction().
  emitDecodeInstruction(OS, IsVarLenInst, TableInfo);

  OS << "\n} // namespace\n";
}

void llvm::EmitDecoder(const RecordKeeper &RK, raw_ostream &OS) {
  DecoderEmitter(RK).run(OS);
}
