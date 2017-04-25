//===- YAMLOutputStyle.cpp ------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YAMLOutputStyle.h"

#include "PdbYaml.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/ModuleSubstream.h"
#include "llvm/DebugInfo/CodeView/ModuleSubstreamVisitor.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"

using namespace llvm;
using namespace llvm::pdb;

YAMLOutputStyle::YAMLOutputStyle(PDBFile &File)
    : File(File), Out(outs()), Obj(File.getAllocator()) {
  Out.setWriteDefaultValues(!opts::pdb2yaml::Minimal);
}

Error YAMLOutputStyle::dump() {
  if (opts::pdb2yaml::StreamDirectory)
    opts::pdb2yaml::StreamMetadata = true;
  if (opts::pdb2yaml::DbiModuleSyms)
    opts::pdb2yaml::DbiModuleInfo = true;

  if (opts::pdb2yaml::DbiModuleSourceLineInfo)
    opts::pdb2yaml::DbiModuleSourceFileInfo = true;

  if (opts::pdb2yaml::DbiModuleSourceFileInfo)
    opts::pdb2yaml::DbiModuleInfo = true;

  if (opts::pdb2yaml::DbiModuleInfo)
    opts::pdb2yaml::DbiStream = true;

  if (auto EC = dumpFileHeaders())
    return EC;

  if (auto EC = dumpStreamMetadata())
    return EC;

  if (auto EC = dumpStreamDirectory())
    return EC;

  if (auto EC = dumpStringTable())
    return EC;

  if (auto EC = dumpPDBStream())
    return EC;

  if (auto EC = dumpDbiStream())
    return EC;

  if (auto EC = dumpTpiStream())
    return EC;

  if (auto EC = dumpIpiStream())
    return EC;

  flush();
  return Error::success();
}

namespace {
class C13SubstreamVisitor : public codeview::IModuleSubstreamVisitor {
public:
  C13SubstreamVisitor(llvm::pdb::yaml::PdbSourceFileInfo &Info, PDBFile &F)
      : Info(Info), F(F) {}

  Error visitUnknown(codeview::ModuleSubstreamKind Kind,
                     BinaryStreamRef Stream) override {
    return Error::success();
  }

  Error
  visitFileChecksums(BinaryStreamRef Data,
                     const codeview::FileChecksumArray &Checksums) override {
    for (const auto &C : Checksums) {
      llvm::pdb::yaml::PdbSourceFileChecksumEntry Entry;
      if (auto Result = getGlobalString(C.FileNameOffset))
        Entry.FileName = *Result;
      else
        return Result.takeError();

      Entry.Kind = C.Kind;
      Entry.ChecksumBytes.Bytes = C.Checksum;
      Info.FileChecksums.push_back(Entry);
    }
    return Error::success();
  }

  Error visitLines(BinaryStreamRef Data,
                   const codeview::LineSubstreamHeader *Header,
                   const codeview::LineInfoArray &Lines) override {

    Info.Lines.CodeSize = Header->CodeSize;
    Info.Lines.Flags =
        static_cast<codeview::LineFlags>(uint16_t(Header->Flags));
    Info.Lines.RelocOffset = Header->RelocOffset;
    Info.Lines.RelocSegment = Header->RelocSegment;

    for (const auto &L : Lines) {
      llvm::pdb::yaml::PdbSourceLineBlock Block;

      if (auto Result = getDbiFileName(L.NameIndex))
        Block.FileName = *Result;
      else
        return Result.takeError();

      for (const auto &N : L.LineNumbers) {
        llvm::pdb::yaml::PdbSourceLineEntry Line;
        Line.Offset = N.Offset;
        codeview::LineInfo LI(N.Flags);
        Line.LineStart = LI.getStartLine();
        Line.EndDelta = LI.getEndLine();
        Line.IsStatement = LI.isStatement();
        Block.Lines.push_back(Line);
      }

      if (Info.Lines.Flags & codeview::LineFlags::HaveColumns) {
        for (const auto &C : L.Columns) {
          llvm::pdb::yaml::PdbSourceColumnEntry Column;
          Column.StartColumn = C.StartColumn;
          Column.EndColumn = C.EndColumn;
          Block.Columns.push_back(Column);
        }
      }

      Info.Lines.LineInfo.push_back(Block);
    }
    return Error::success();
  }

private:
  Expected<StringRef> getGlobalString(uint32_t Offset) {
    auto ST = F.getStringTable();
    if (!ST)
      return ST.takeError();

    return ST->getStringForID(Offset);
  }
  Expected<StringRef> getDbiFileName(uint32_t Offset) {
    auto DS = F.getPDBDbiStream();
    if (!DS)
      return DS.takeError();
    return DS->getFileNameForIndex(Offset);
  }

  llvm::pdb::yaml::PdbSourceFileInfo &Info;
  PDBFile &F;
};
}

Expected<Optional<llvm::pdb::yaml::PdbSourceFileInfo>>
YAMLOutputStyle::getFileLineInfo(const pdb::ModStream &ModS) {
  if (!ModS.hasLineInfo())
    return None;

  yaml::PdbSourceFileInfo Info;
  bool Error = false;
  C13SubstreamVisitor Visitor(Info, File);
  for (auto &Substream : ModS.lines(&Error)) {
    if (auto E = codeview::visitModuleSubstream(Substream, Visitor))
      return std::move(E);
  }

  return Info;
}

Error YAMLOutputStyle::dumpFileHeaders() {
  if (opts::pdb2yaml::NoFileHeaders)
    return Error::success();

  yaml::MSFHeaders Headers;
  Obj.Headers.emplace();
  Obj.Headers->SuperBlock.NumBlocks = File.getBlockCount();
  Obj.Headers->SuperBlock.BlockMapAddr = File.getBlockMapIndex();
  Obj.Headers->SuperBlock.BlockSize = File.getBlockSize();
  auto Blocks = File.getDirectoryBlockArray();
  Obj.Headers->DirectoryBlocks.assign(Blocks.begin(), Blocks.end());
  Obj.Headers->NumDirectoryBlocks = File.getNumDirectoryBlocks();
  Obj.Headers->SuperBlock.NumDirectoryBytes = File.getNumDirectoryBytes();
  Obj.Headers->NumStreams =
      opts::pdb2yaml::StreamMetadata ? File.getNumStreams() : 0;
  Obj.Headers->SuperBlock.FreeBlockMapBlock = File.getFreeBlockMapBlock();
  Obj.Headers->SuperBlock.Unknown1 = File.getUnknown1();
  Obj.Headers->FileSize = File.getFileSize();

  return Error::success();
}

Error YAMLOutputStyle::dumpStringTable() {
  if (!opts::pdb2yaml::StringTable)
    return Error::success();

  Obj.StringTable.emplace();
  auto ExpectedST = File.getStringTable();
  if (!ExpectedST)
    return ExpectedST.takeError();

  const auto &ST = ExpectedST.get();
  for (auto ID : ST.name_ids()) {
    StringRef S = ST.getStringForID(ID);
    if (!S.empty())
      Obj.StringTable->push_back(S);
  }
  return Error::success();
}

Error YAMLOutputStyle::dumpStreamMetadata() {
  if (!opts::pdb2yaml::StreamMetadata)
    return Error::success();

  Obj.StreamSizes.emplace();
  Obj.StreamSizes->assign(File.getStreamSizes().begin(),
                          File.getStreamSizes().end());
  return Error::success();
}

Error YAMLOutputStyle::dumpStreamDirectory() {
  if (!opts::pdb2yaml::StreamDirectory)
    return Error::success();

  auto StreamMap = File.getStreamMap();
  Obj.StreamMap.emplace();
  for (auto &Stream : StreamMap) {
    pdb::yaml::StreamBlockList BlockList;
    BlockList.Blocks.assign(Stream.begin(), Stream.end());
    Obj.StreamMap->push_back(BlockList);
  }

  return Error::success();
}

Error YAMLOutputStyle::dumpPDBStream() {
  if (!opts::pdb2yaml::PdbStream)
    return Error::success();

  auto IS = File.getPDBInfoStream();
  if (!IS)
    return IS.takeError();

  auto &InfoS = IS.get();
  Obj.PdbStream.emplace();
  Obj.PdbStream->Age = InfoS.getAge();
  Obj.PdbStream->Guid = InfoS.getGuid();
  Obj.PdbStream->Signature = InfoS.getSignature();
  Obj.PdbStream->Version = InfoS.getVersion();
  Obj.PdbStream->Features = InfoS.getFeatureSignatures();

  return Error::success();
}

Error YAMLOutputStyle::dumpDbiStream() {
  if (!opts::pdb2yaml::DbiStream)
    return Error::success();

  auto DbiS = File.getPDBDbiStream();
  if (!DbiS)
    return DbiS.takeError();

  auto &DS = DbiS.get();
  Obj.DbiStream.emplace();
  Obj.DbiStream->Age = DS.getAge();
  Obj.DbiStream->BuildNumber = DS.getBuildNumber();
  Obj.DbiStream->Flags = DS.getFlags();
  Obj.DbiStream->MachineType = DS.getMachineType();
  Obj.DbiStream->PdbDllRbld = DS.getPdbDllRbld();
  Obj.DbiStream->PdbDllVersion = DS.getPdbDllVersion();
  Obj.DbiStream->VerHeader = DS.getDbiVersion();
  if (opts::pdb2yaml::DbiModuleInfo) {
    for (const auto &MI : DS.modules()) {
      yaml::PdbDbiModuleInfo DMI;
      DMI.Mod = MI.Info.getModuleName();
      DMI.Obj = MI.Info.getObjFileName();
      if (opts::pdb2yaml::DbiModuleSourceFileInfo)
        DMI.SourceFiles = MI.SourceFiles;

      auto ModStreamData = msf::MappedBlockStream::createIndexedStream(
          File.getMsfLayout(), File.getMsfBuffer(),
          MI.Info.getModuleStreamIndex());

      pdb::ModStream ModS(MI.Info, std::move(ModStreamData));
      if (auto EC = ModS.reload())
        return EC;

      if (opts::pdb2yaml::DbiModuleSourceLineInfo) {
        auto ExpectedInfo = getFileLineInfo(ModS);
        if (!ExpectedInfo)
          return ExpectedInfo.takeError();
        DMI.FileLineInfo = *ExpectedInfo;
      }

      if (opts::pdb2yaml::DbiModuleSyms &&
          MI.Info.getModuleStreamIndex() != kInvalidStreamIndex) {
        DMI.Modi.emplace();

        DMI.Modi->Signature = ModS.signature();
        bool HadError = false;
        for (auto &Sym : ModS.symbols(&HadError)) {
          pdb::yaml::PdbSymbolRecord Record{Sym};
          DMI.Modi->Symbols.push_back(Record);
        }
      }
      Obj.DbiStream->ModInfos.push_back(DMI);
    }
  }
  return Error::success();
}

Error YAMLOutputStyle::dumpTpiStream() {
  if (!opts::pdb2yaml::TpiStream)
    return Error::success();

  auto TpiS = File.getPDBTpiStream();
  if (!TpiS)
    return TpiS.takeError();

  auto &TS = TpiS.get();
  Obj.TpiStream.emplace();
  Obj.TpiStream->Version = TS.getTpiVersion();
  for (auto &Record : TS.types(nullptr)) {
    yaml::PdbTpiRecord R;
    // It's not necessary to set R.RecordData here.  That only exists as a
    // way to have the `PdbTpiRecord` structure own the memory that `R.Record`
    // references.  In the case of reading an existing PDB though, that memory
    // is owned by the backing stream.
    R.Record = Record;
    Obj.TpiStream->Records.push_back(R);
  }

  return Error::success();
}

Error YAMLOutputStyle::dumpIpiStream() {
  if (!opts::pdb2yaml::IpiStream)
    return Error::success();

  auto IpiS = File.getPDBIpiStream();
  if (!IpiS)
    return IpiS.takeError();

  auto &IS = IpiS.get();
  Obj.IpiStream.emplace();
  Obj.IpiStream->Version = IS.getTpiVersion();
  for (auto &Record : IS.types(nullptr)) {
    yaml::PdbTpiRecord R;
    R.Record = Record;
    Obj.IpiStream->Records.push_back(R);
  }

  return Error::success();
}

void YAMLOutputStyle::flush() {
  Out << Obj;
  outs().flush();
}
