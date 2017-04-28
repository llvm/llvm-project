//===- PdbYAML.cpp -------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PdbYaml.h"

#include "YamlSerializationContext.h"
#include "YamlSymbolDumper.h"
#include "YamlTypeDumper.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolSerializer.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeSerializer.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

using namespace llvm;
using namespace llvm::pdb;
using namespace llvm::pdb::yaml;
using namespace llvm::yaml;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint32_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::StringRef)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::NamedStreamMapping)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbDbiModuleInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbSourceFileChecksumEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbSourceLineEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbSourceColumnEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbSourceLineBlock)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbSymbolRecord)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbTpiRecord)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::StreamBlockList)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(llvm::pdb::PdbRaw_FeatureSig)

namespace llvm {
namespace yaml {

template <> struct ScalarTraits<llvm::pdb::PDB_UniqueId> {
  static void output(const llvm::pdb::PDB_UniqueId &S, void *,
                     llvm::raw_ostream &OS) {
    OS << S;
  }

  static StringRef input(StringRef Scalar, void *Ctx,
                         llvm::pdb::PDB_UniqueId &S) {
    if (Scalar.size() != 38)
      return "GUID strings are 38 characters long";
    if (Scalar[0] != '{' || Scalar[37] != '}')
      return "GUID is not enclosed in {}";
    if (Scalar[9] != '-' || Scalar[14] != '-' || Scalar[19] != '-' ||
        Scalar[24] != '-')
      return "GUID sections are not properly delineated with dashes";

    uint8_t *OutBuffer = S.Guid;
    for (auto Iter = Scalar.begin(); Iter != Scalar.end();) {
      if (*Iter == '-' || *Iter == '{' || *Iter == '}') {
        ++Iter;
        continue;
      }
      uint8_t Value = (llvm::hexDigitValue(*Iter) << 4);
      ++Iter;
      Value |= llvm::hexDigitValue(*Iter);
      ++Iter;
      *OutBuffer++ = Value;
    }

    return "";
  }

  static bool mustQuote(StringRef Scalar) { return needsQuotes(Scalar); }
};

template <> struct ScalarEnumerationTraits<llvm::pdb::PDB_Machine> {
  static void enumeration(IO &io, llvm::pdb::PDB_Machine &Value) {
    io.enumCase(Value, "Invalid", PDB_Machine::Invalid);
    io.enumCase(Value, "Am33", PDB_Machine::Am33);
    io.enumCase(Value, "Amd64", PDB_Machine::Amd64);
    io.enumCase(Value, "Arm", PDB_Machine::Arm);
    io.enumCase(Value, "ArmNT", PDB_Machine::ArmNT);
    io.enumCase(Value, "Ebc", PDB_Machine::Ebc);
    io.enumCase(Value, "x86", PDB_Machine::x86);
    io.enumCase(Value, "Ia64", PDB_Machine::Ia64);
    io.enumCase(Value, "M32R", PDB_Machine::M32R);
    io.enumCase(Value, "Mips16", PDB_Machine::Mips16);
    io.enumCase(Value, "MipsFpu", PDB_Machine::MipsFpu);
    io.enumCase(Value, "MipsFpu16", PDB_Machine::MipsFpu16);
    io.enumCase(Value, "PowerPCFP", PDB_Machine::PowerPCFP);
    io.enumCase(Value, "R4000", PDB_Machine::R4000);
    io.enumCase(Value, "SH3", PDB_Machine::SH3);
    io.enumCase(Value, "SH3DSP", PDB_Machine::SH3DSP);
    io.enumCase(Value, "Thumb", PDB_Machine::Thumb);
    io.enumCase(Value, "WceMipsV2", PDB_Machine::WceMipsV2);
  }
};

template <> struct ScalarEnumerationTraits<llvm::pdb::PdbRaw_DbiVer> {
  static void enumeration(IO &io, llvm::pdb::PdbRaw_DbiVer &Value) {
    io.enumCase(Value, "V41", llvm::pdb::PdbRaw_DbiVer::PdbDbiVC41);
    io.enumCase(Value, "V50", llvm::pdb::PdbRaw_DbiVer::PdbDbiV50);
    io.enumCase(Value, "V60", llvm::pdb::PdbRaw_DbiVer::PdbDbiV60);
    io.enumCase(Value, "V70", llvm::pdb::PdbRaw_DbiVer::PdbDbiV70);
    io.enumCase(Value, "V110", llvm::pdb::PdbRaw_DbiVer::PdbDbiV110);
  }
};

template <> struct ScalarEnumerationTraits<llvm::pdb::PdbRaw_ImplVer> {
  static void enumeration(IO &io, llvm::pdb::PdbRaw_ImplVer &Value) {
    io.enumCase(Value, "VC2", llvm::pdb::PdbRaw_ImplVer::PdbImplVC2);
    io.enumCase(Value, "VC4", llvm::pdb::PdbRaw_ImplVer::PdbImplVC4);
    io.enumCase(Value, "VC41", llvm::pdb::PdbRaw_ImplVer::PdbImplVC41);
    io.enumCase(Value, "VC50", llvm::pdb::PdbRaw_ImplVer::PdbImplVC50);
    io.enumCase(Value, "VC98", llvm::pdb::PdbRaw_ImplVer::PdbImplVC98);
    io.enumCase(Value, "VC70Dep", llvm::pdb::PdbRaw_ImplVer::PdbImplVC70Dep);
    io.enumCase(Value, "VC70", llvm::pdb::PdbRaw_ImplVer::PdbImplVC70);
    io.enumCase(Value, "VC80", llvm::pdb::PdbRaw_ImplVer::PdbImplVC80);
    io.enumCase(Value, "VC110", llvm::pdb::PdbRaw_ImplVer::PdbImplVC110);
    io.enumCase(Value, "VC140", llvm::pdb::PdbRaw_ImplVer::PdbImplVC140);
  }
};

template <> struct ScalarEnumerationTraits<llvm::pdb::PdbRaw_TpiVer> {
  static void enumeration(IO &io, llvm::pdb::PdbRaw_TpiVer &Value) {
    io.enumCase(Value, "VC40", llvm::pdb::PdbRaw_TpiVer::PdbTpiV40);
    io.enumCase(Value, "VC41", llvm::pdb::PdbRaw_TpiVer::PdbTpiV41);
    io.enumCase(Value, "VC50", llvm::pdb::PdbRaw_TpiVer::PdbTpiV50);
    io.enumCase(Value, "VC70", llvm::pdb::PdbRaw_TpiVer::PdbTpiV70);
    io.enumCase(Value, "VC80", llvm::pdb::PdbRaw_TpiVer::PdbTpiV80);
  }
};

template <> struct ScalarEnumerationTraits<llvm::pdb::PdbRaw_FeatureSig> {
  static void enumeration(IO &io, PdbRaw_FeatureSig &Features) {
    io.enumCase(Features, "MinimalDebugInfo",
                PdbRaw_FeatureSig::MinimalDebugInfo);
    io.enumCase(Features, "NoTypeMerge", PdbRaw_FeatureSig::NoTypeMerge);
    io.enumCase(Features, "VC110", PdbRaw_FeatureSig::VC110);
    io.enumCase(Features, "VC140", PdbRaw_FeatureSig::VC140);
  }
};

template <> struct ScalarEnumerationTraits<llvm::codeview::FileChecksumKind> {
  static void enumeration(IO &io, llvm::codeview::FileChecksumKind &Kind) {
    io.enumCase(Kind, "None", llvm::codeview::FileChecksumKind::None);
    io.enumCase(Kind, "MD5", llvm::codeview::FileChecksumKind::MD5);
    io.enumCase(Kind, "SHA1", llvm::codeview::FileChecksumKind::SHA1);
    io.enumCase(Kind, "SHA256", llvm::codeview::FileChecksumKind::SHA256);
  }
};

template <> struct ScalarBitSetTraits<llvm::codeview::LineFlags> {
  static void bitset(IO &io, llvm::codeview::LineFlags &Flags) {
    io.bitSetCase(Flags, "HasColumnInfo",
                  llvm::codeview::LineFlags::HaveColumns);
    io.enumFallback<Hex16>(Flags);
  }
};
}
}

void ScalarTraits<HexFormattedString>::output(const HexFormattedString &Value,
                                              void *ctx, raw_ostream &Out) {
  StringRef Bytes(reinterpret_cast<const char *>(Value.Bytes.data()),
                  Value.Bytes.size());
  Out << toHex(Bytes);
}

StringRef ScalarTraits<HexFormattedString>::input(StringRef Scalar, void *ctxt,
                                                  HexFormattedString &Value) {
  std::string H = fromHex(Scalar);
  Value.Bytes.assign(H.begin(), H.end());
  return StringRef();
}

void MappingTraits<PdbObject>::mapping(IO &IO, PdbObject &Obj) {
  // Create a single serialization context that will be passed through the
  // entire process of serializing / deserializing a Tpi Stream.  This is
  // especially important when we are going from Pdb -> Yaml because we need
  // to maintain state in a TypeTableBuilder across mappings, and at the end of
  // the entire process, we need to have one TypeTableBuilder that has every
  // record.
  pdb::yaml::SerializationContext Context(IO, Obj.Allocator);


  IO.mapOptional("MSF", Obj.Headers);
  IO.mapOptional("StreamSizes", Obj.StreamSizes);
  IO.mapOptional("StreamMap", Obj.StreamMap);
  IO.mapOptional("StringTable", Obj.StringTable);
  IO.mapOptional("PdbStream", Obj.PdbStream);
  IO.mapOptionalWithContext("DbiStream", Obj.DbiStream, Context);
  IO.mapOptionalWithContext("TpiStream", Obj.TpiStream, Context);
  IO.mapOptionalWithContext("IpiStream", Obj.IpiStream, Context);
}

void MappingTraits<MSFHeaders>::mapping(IO &IO, MSFHeaders &Obj) {
  IO.mapOptional("SuperBlock", Obj.SuperBlock);
  IO.mapOptional("NumDirectoryBlocks", Obj.NumDirectoryBlocks);
  IO.mapOptional("DirectoryBlocks", Obj.DirectoryBlocks);
  IO.mapOptional("NumStreams", Obj.NumStreams);
  IO.mapOptional("FileSize", Obj.FileSize);
}

void MappingTraits<msf::SuperBlock>::mapping(IO &IO, msf::SuperBlock &SB) {
  if (!IO.outputting()) {
    ::memcpy(SB.MagicBytes, msf::Magic, sizeof(msf::Magic));
  }

  using u32 = support::ulittle32_t;
  IO.mapOptional("BlockSize", SB.BlockSize, u32(4096U));
  IO.mapOptional("FreeBlockMap", SB.FreeBlockMapBlock, u32(0U));
  IO.mapOptional("NumBlocks", SB.NumBlocks, u32(0U));
  IO.mapOptional("NumDirectoryBytes", SB.NumDirectoryBytes, u32(0U));
  IO.mapOptional("Unknown1", SB.Unknown1, u32(0U));
  IO.mapOptional("BlockMapAddr", SB.BlockMapAddr, u32(0U));
}

void MappingTraits<StreamBlockList>::mapping(IO &IO, StreamBlockList &SB) {
  IO.mapRequired("Stream", SB.Blocks);
}

void MappingTraits<PdbInfoStream>::mapping(IO &IO, PdbInfoStream &Obj) {
  IO.mapOptional("Age", Obj.Age, 1U);
  IO.mapOptional("Guid", Obj.Guid);
  IO.mapOptional("Signature", Obj.Signature, 0U);
  IO.mapOptional("Features", Obj.Features);
  IO.mapOptional("Version", Obj.Version, PdbImplVC70);
}

void MappingContextTraits<PdbDbiStream, pdb::yaml::SerializationContext>::mapping(IO &IO, PdbDbiStream &Obj, pdb::yaml::SerializationContext &Context) {
  IO.mapOptional("VerHeader", Obj.VerHeader, PdbDbiV70);
  IO.mapOptional("Age", Obj.Age, 1U);
  IO.mapOptional("BuildNumber", Obj.BuildNumber, uint16_t(0U));
  IO.mapOptional("PdbDllVersion", Obj.PdbDllVersion, 0U);
  IO.mapOptional("PdbDllRbld", Obj.PdbDllRbld, uint16_t(0U));
  IO.mapOptional("Flags", Obj.Flags, uint16_t(1U));
  IO.mapOptional("MachineType", Obj.MachineType, PDB_Machine::x86);
  IO.mapOptionalWithContext("Modules", Obj.ModInfos, Context);
}

void MappingContextTraits<PdbTpiStream, pdb::yaml::SerializationContext>::mapping(
    IO &IO, pdb::yaml::PdbTpiStream &Obj, pdb::yaml::SerializationContext &Context) {
  IO.mapOptional("Version", Obj.Version, PdbTpiV80);
  IO.mapRequired("Records", Obj.Records, Context);
}

void MappingTraits<NamedStreamMapping>::mapping(IO &IO,
                                                NamedStreamMapping &Obj) {
  IO.mapRequired("Name", Obj.StreamName);
  IO.mapRequired("StreamNum", Obj.StreamNumber);
}

void MappingContextTraits<PdbSymbolRecord, pdb::yaml::SerializationContext>::mapping(IO &IO, PdbSymbolRecord &Obj, pdb::yaml::SerializationContext &Context) {
  codeview::SymbolVisitorCallbackPipeline Pipeline;
  codeview::SymbolSerializer Serializer(Context.Allocator);
  codeview::SymbolDeserializer Deserializer(nullptr);
  codeview::yaml::YamlSymbolDumper Dumper(IO);

  if (IO.outputting()) {
    // For PDB to Yaml, deserialize into a high level record type, then dump it.
    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Dumper);
  } else {
    // For the other way around, dump it into a concrete structure, and then
    // serialize it into the CVRecord.
    Pipeline.addCallbackToPipeline(Dumper);
    Pipeline.addCallbackToPipeline(Serializer);
  }

  codeview::CVSymbolVisitor Visitor(Pipeline);
  consumeError(Visitor.visitSymbolRecord(Obj.Record));
}

void MappingContextTraits<PdbModiStream, pdb::yaml::SerializationContext>::mapping(IO &IO, PdbModiStream &Obj, pdb::yaml::SerializationContext &Context) {
  IO.mapOptional("Signature", Obj.Signature, 4U);
  IO.mapRequired("Records", Obj.Symbols, Context);
}

void MappingContextTraits<PdbDbiModuleInfo, pdb::yaml::SerializationContext>::mapping(IO &IO, PdbDbiModuleInfo &Obj, pdb::yaml::SerializationContext &Context) {
  IO.mapRequired("Module", Obj.Mod);
  IO.mapOptional("ObjFile", Obj.Obj, Obj.Mod);
  IO.mapOptional("SourceFiles", Obj.SourceFiles);
  IO.mapOptionalWithContext("LineInfo", Obj.FileLineInfo, Context);
  IO.mapOptionalWithContext("Modi", Obj.Modi, Context);
}

void MappingContextTraits<pdb::yaml::PdbSourceLineEntry,
                          pdb::yaml::SerializationContext>::
    mapping(IO &IO, PdbSourceLineEntry &Obj,
            pdb::yaml::SerializationContext &Context) {
  IO.mapRequired("Offset", Obj.Offset);
  IO.mapRequired("LineStart", Obj.LineStart);
  IO.mapRequired("IsStatement", Obj.IsStatement);
  IO.mapRequired("EndDelta", Obj.EndDelta);
}

void MappingContextTraits<pdb::yaml::PdbSourceColumnEntry,
                          pdb::yaml::SerializationContext>::
    mapping(IO &IO, PdbSourceColumnEntry &Obj,
            pdb::yaml::SerializationContext &Context) {
  IO.mapRequired("StartColumn", Obj.StartColumn);
  IO.mapRequired("EndColumn", Obj.EndColumn);
}

void MappingContextTraits<pdb::yaml::PdbSourceLineBlock,
                          pdb::yaml::SerializationContext>::
    mapping(IO &IO, PdbSourceLineBlock &Obj,
            pdb::yaml::SerializationContext &Context) {
  IO.mapRequired("FileName", Obj.FileName);
  IO.mapRequired("Lines", Obj.Lines, Context);
  IO.mapRequired("Columns", Obj.Columns, Context);
}

void MappingContextTraits<pdb::yaml::PdbSourceFileChecksumEntry,
                          pdb::yaml::SerializationContext>::
    mapping(IO &IO, PdbSourceFileChecksumEntry &Obj,
            pdb::yaml::SerializationContext &Context) {
  IO.mapRequired("FileName", Obj.FileName);
  IO.mapRequired("Kind", Obj.Kind);
  IO.mapRequired("Checksum", Obj.ChecksumBytes);
}

void MappingContextTraits<pdb::yaml::PdbSourceLineInfo,
                          pdb::yaml::SerializationContext>::
    mapping(IO &IO, PdbSourceLineInfo &Obj,
            pdb::yaml::SerializationContext &Context) {
  IO.mapRequired("CodeSize", Obj.CodeSize);
  IO.mapRequired("Flags", Obj.Flags);
  IO.mapRequired("RelocOffset", Obj.RelocOffset);
  IO.mapRequired("RelocSegment", Obj.RelocSegment);
  IO.mapRequired("LineInfo", Obj.LineInfo, Context);
}

void MappingContextTraits<pdb::yaml::PdbSourceFileInfo,
                          pdb::yaml::SerializationContext>::
    mapping(IO &IO, PdbSourceFileInfo &Obj,
            pdb::yaml::SerializationContext &Context) {
  IO.mapOptionalWithContext("Lines", Obj.Lines, Context);
  IO.mapOptionalWithContext("Checksums", Obj.FileChecksums, Context);
}

void MappingContextTraits<PdbTpiRecord, pdb::yaml::SerializationContext>::
    mapping(IO &IO, pdb::yaml::PdbTpiRecord &Obj,
            pdb::yaml::SerializationContext &Context) {
  codeview::TypeVisitorCallbackPipeline Pipeline;
  codeview::TypeDeserializer Deserializer;
  codeview::TypeSerializer Serializer(Context.Allocator);
  pdb::TpiHashUpdater Hasher;

  if (IO.outputting()) {
    // For PDB to Yaml, deserialize into a high level record type, then dump it.
    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Context.Dumper);
  } else {
    // For Yaml to PDB, extract from the high level record type, then write it
    // to bytes.

    // This might be interpreted as a hack, but serializing FieldList
    // sub-records requires having access to the same serializer being used by
    // the FieldList itself.
    Context.ActiveSerializer = &Serializer;
    Pipeline.addCallbackToPipeline(Context.Dumper);
    Pipeline.addCallbackToPipeline(Serializer);
    Pipeline.addCallbackToPipeline(Hasher);
  }

  codeview::CVTypeVisitor Visitor(Pipeline);
  consumeError(Visitor.visitTypeRecord(Obj.Record));
  Context.ActiveSerializer = nullptr;
}
