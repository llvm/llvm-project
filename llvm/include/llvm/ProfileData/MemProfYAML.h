#ifndef LLVM_PROFILEDATA_MEMPROFYAML_H_
#define LLVM_PROFILEDATA_MEMPROFYAML_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace memprof {
// A "typedef" for GUID.  See ScalarTraits<memprof::GUIDHex64> for how a GUID is
// serialized and deserialized in YAML.
LLVM_YAML_STRONG_TYPEDEF(uint64_t, GUIDHex64)

// Helper struct for AllMemProfData.  In YAML, we treat the GUID and the fields
// within MemProfRecord at the same level as if the GUID were part of
// MemProfRecord.
struct GUIDMemProfRecordPair {
  GUIDHex64 GUID;
  MemProfRecord Record;
};

// The top-level data structure, only used with YAML for now.
struct AllMemProfData {
  std::vector<GUIDMemProfRecordPair> HeapProfileRecords;
};
} // namespace memprof

namespace yaml {
template <> struct ScalarTraits<memprof::GUIDHex64> {
  static void output(const memprof::GUIDHex64 &Val, void *, raw_ostream &Out) {
    // Print GUID as a hexadecimal number with 0x prefix, no padding to keep
    // test strings compact.
    Out << format("0x%" PRIx64, (uint64_t)Val);
  }
  static StringRef input(StringRef Scalar, void *, memprof::GUIDHex64 &Val) {
    // Reject decimal GUIDs.
    if (all_of(Scalar, [](char C) { return std::isdigit(C); }))
      return "use a hexadecimal GUID or a function instead";

    uint64_t Num;
    if (Scalar.starts_with_insensitive("0x")) {
      // Accept hexadecimal numbers starting with 0x or 0X.
      if (Scalar.getAsInteger(0, Num))
        return "invalid hex64 number";
      Val = Num;
    } else {
      // Otherwise, treat the input as a string containing a function name.
      Val = memprof::getGUID(Scalar);
    }
    return StringRef();
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

template <> struct MappingTraits<memprof::Frame> {
  // Essentially the same as memprof::Frame except that Function is of type
  // memprof::GUIDHex64 instead of GlobalValue::GUID.  This class helps in two
  // ways.  During serialization, we print Function as a 16-digit hexadecimal
  // number.  During deserialization, we accept a function name as an
  // alternative to the usual GUID expressed as a hexadecimal number.
  class FrameWithHex64 {
  public:
    FrameWithHex64(IO &) {}
    FrameWithHex64(IO &, const memprof::Frame &F)
        : Function(F.Function), LineOffset(F.LineOffset), Column(F.Column),
          IsInlineFrame(F.IsInlineFrame) {}
    memprof::Frame denormalize(IO &) {
      return memprof::Frame(Function, LineOffset, Column, IsInlineFrame);
    }

    memprof::GUIDHex64 Function = 0;
    static_assert(std::is_same_v<decltype(Function.value),
                                 decltype(memprof::Frame::Function)>);
    decltype(memprof::Frame::LineOffset) LineOffset = 0;
    decltype(memprof::Frame::Column) Column = 0;
    decltype(memprof::Frame::IsInlineFrame) IsInlineFrame = false;
  };

  static void mapping(IO &Io, memprof::Frame &F) {
    MappingNormalization<FrameWithHex64, memprof::Frame> Keys(Io, F);

    Io.mapRequired("Function", Keys->Function);
    Io.mapRequired("LineOffset", Keys->LineOffset);
    Io.mapRequired("Column", Keys->Column);
    Io.mapRequired("IsInlineFrame", Keys->IsInlineFrame);

    // Assert that the definition of Frame matches what we expect.  The
    // structured bindings below detect changes to the number of fields.
    // static_assert checks the type of each field.
    const auto &[Function, SymbolName, LineOffset, Column, IsInlineFrame] = F;
    static_assert(
        std::is_same_v<remove_cvref_t<decltype(Function)>, GlobalValue::GUID>);
    static_assert(std::is_same_v<remove_cvref_t<decltype(SymbolName)>,
                                 std::unique_ptr<std::string>>);
    static_assert(
        std::is_same_v<remove_cvref_t<decltype(LineOffset)>, uint32_t>);
    static_assert(std::is_same_v<remove_cvref_t<decltype(Column)>, uint32_t>);
    static_assert(
        std::is_same_v<remove_cvref_t<decltype(IsInlineFrame)>, bool>);

    // MSVC issues unused variable warnings despite the uses in static_assert
    // above.
    (void)Function;
    (void)SymbolName;
    (void)LineOffset;
    (void)Column;
    (void)IsInlineFrame;
  }

  // Request the inline notation for brevity:
  //   { Function: 123, LineOffset: 11, Column: 10; IsInlineFrame: true }
  static const bool flow = true;
};

template <> struct CustomMappingTraits<memprof::PortableMemInfoBlock> {
  static void inputOne(IO &Io, StringRef KeyStr,
                       memprof::PortableMemInfoBlock &MIB) {
    // PortableMemInfoBlock keeps track of the set of fields that actually have
    // values.  We update the set here as we receive a key-value pair from the
    // YAML document.
    //
    // We set MIB.Name via a temporary variable because ScalarTraits<uintptr_t>
    // isn't available on macOS.
#define MIBEntryDef(NameTag, Name, Type)                                       \
  if (KeyStr == #Name) {                                                       \
    uint64_t Value;                                                            \
    Io.mapRequired(KeyStr.str().c_str(), Value);                               \
    MIB.Name = static_cast<Type>(Value);                                       \
    MIB.Schema.set(llvm::to_underlying(memprof::Meta::Name));                  \
    return;                                                                    \
  }
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
    Io.setError("Key is not a valid validation event");
  }

  static void output(IO &Io, memprof::PortableMemInfoBlock &MIB) {
    auto Schema = MIB.getSchema();
#define MIBEntryDef(NameTag, Name, Type)                                       \
  if (Schema.test(llvm::to_underlying(memprof::Meta::Name))) {                 \
    uint64_t Value = MIB.Name;                                                 \
    Io.mapRequired(#Name, Value);                                              \
  }
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  }
};

template <> struct MappingTraits<memprof::AllocationInfo> {
  static void mapping(IO &Io, memprof::AllocationInfo &AI) {
    Io.mapRequired("Callstack", AI.CallStack);
    Io.mapRequired("MemInfoBlock", AI.Info);
  }
};

// In YAML, we use GUIDMemProfRecordPair instead of MemProfRecord so that we can
// treat the GUID and the fields within MemProfRecord at the same level as if
// the GUID were part of MemProfRecord.
template <> struct MappingTraits<memprof::CallSiteInfo> {
  // Helper class to normalize CalleeGuids to use GUIDHex64 for YAML I/O.
  class CallSiteInfoWithHex64Guids {
  public:
    CallSiteInfoWithHex64Guids(IO &) {}
    CallSiteInfoWithHex64Guids(IO &, const memprof::CallSiteInfo &CS)
        : Frames(CS.Frames) {
      // Convert uint64_t GUIDs to GUIDHex64 for serialization.
      CalleeGuids.reserve(CS.CalleeGuids.size());
      for (uint64_t Guid : CS.CalleeGuids)
        CalleeGuids.push_back(memprof::GUIDHex64(Guid));
    }

    memprof::CallSiteInfo denormalize(IO &) {
      memprof::CallSiteInfo CS;
      CS.Frames = Frames;
      // Convert GUIDHex64 back to uint64_t GUIDs after deserialization.
      CS.CalleeGuids.reserve(CalleeGuids.size());
      for (memprof::GUIDHex64 HexGuid : CalleeGuids)
        CS.CalleeGuids.push_back(HexGuid.value);
      return CS;
    }

    // Keep Frames as is, since MappingTraits<memprof::Frame> handles its
    // Function GUID.
    decltype(memprof::CallSiteInfo::Frames) Frames;
    // Use a vector of GUIDHex64 for CalleeGuids to leverage its ScalarTraits.
    SmallVector<memprof::GUIDHex64> CalleeGuids;
  };

  static void mapping(IO &Io, memprof::CallSiteInfo &CS) {
    // Use MappingNormalization to handle the conversion between
    // memprof::CallSiteInfo and CallSiteInfoWithHex64Guids.
    MappingNormalization<CallSiteInfoWithHex64Guids, memprof::CallSiteInfo>
        Keys(Io, CS);
    Io.mapRequired("Frames", Keys->Frames);
    // Map the normalized CalleeGuids (which are now GUIDHex64).
    Io.mapOptional("CalleeGuids", Keys->CalleeGuids);
  }
};

template <> struct MappingTraits<memprof::GUIDMemProfRecordPair> {
  static void mapping(IO &Io, memprof::GUIDMemProfRecordPair &Pair) {
    Io.mapRequired("GUID", Pair.GUID);
    Io.mapRequired("AllocSites", Pair.Record.AllocSites);
    Io.mapRequired("CallSites", Pair.Record.CallSites);
  }
};

template <> struct MappingTraits<memprof::AllMemProfData> {
  static void mapping(IO &Io, memprof::AllMemProfData &Data) {
    Io.mapRequired("HeapProfileRecords", Data.HeapProfileRecords);
  }
};

template <> struct SequenceTraits<SmallVector<memprof::GUIDHex64>> {
  static size_t size(IO &io, SmallVector<memprof::GUIDHex64> &Seq) {
    return Seq.size();
  }
  static memprof::GUIDHex64 &
  element(IO &io, SmallVector<memprof::GUIDHex64> &Seq, size_t Index) {
    if (Index >= Seq.size())
      Seq.resize(Index + 1);
    return Seq[Index];
  }
  static const bool flow = true;
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(memprof::Frame)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<memprof::Frame>)
LLVM_YAML_IS_SEQUENCE_VECTOR(memprof::AllocationInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(memprof::CallSiteInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(memprof::GUIDMemProfRecordPair)
LLVM_YAML_IS_SEQUENCE_VECTOR(memprof::GUIDHex64) // Used for CalleeGuids

#endif // LLVM_PROFILEDATA_MEMPROFYAML_H_
