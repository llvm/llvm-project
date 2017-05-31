//===- YAMLOutputStyle.cpp ------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YAMLOutputStyle.h"

#include "C13DebugFragmentVisitor.h"
#include "PdbYaml.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/CodeView/DebugChecksumsSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugInlineeLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugUnknownSubsection.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/ObjectYAML/CodeViewYAML.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

YAMLOutputStyle::YAMLOutputStyle(PDBFile &File)
    : File(File), Out(outs()), Obj(File.getAllocator()) {
  Out.setWriteDefaultValues(!opts::pdb2yaml::Minimal);
}

Error YAMLOutputStyle::dump() {
  if (opts::pdb2yaml::All) {
    opts::pdb2yaml::StreamMetadata = true;
    opts::pdb2yaml::StreamDirectory = true;
    opts::pdb2yaml::PdbStream = true;
    opts::pdb2yaml::StringTable = true;
    opts::pdb2yaml::DbiStream = true;
    opts::pdb2yaml::DbiModuleInfo = true;
    opts::pdb2yaml::DbiModuleSyms = true;
    opts::pdb2yaml::DbiModuleSourceFileInfo = true;
    opts::pdb2yaml::DbiModuleSourceLineInfo = true;
    opts::pdb2yaml::TpiStream = true;
    opts::pdb2yaml::IpiStream = true;
  }

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

  // Some names from the module source file info get pulled from the string
  // table, so if we're writing module source info, we have to write the string
  // table as well.
  if (opts::pdb2yaml::DbiModuleSourceLineInfo)
    opts::pdb2yaml::StringTable = true;

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
class C13YamlVisitor : public C13DebugFragmentVisitor {
public:
  C13YamlVisitor(CodeViewYAML::SourceFileInfo &Info, PDBFile &F)
      : C13DebugFragmentVisitor(F), Info(Info) {}

  Error handleFileChecksums() override {
    for (const auto &C : *Checksums) {
      CodeViewYAML::SourceFileChecksumEntry Entry;
      if (auto Result = getNameFromStringTable(C.FileNameOffset))
        Entry.FileName = *Result;
      else
        return Result.takeError();

      Entry.Kind = C.Kind;
      Entry.ChecksumBytes.Bytes = C.Checksum;
      Info.FileChecksums.push_back(Entry);
    }
    return Error::success();
  }

  Error handleLines() override {
    for (const auto &LF : Lines) {
      Info.LineFragments.emplace_back();
      auto &Fragment = Info.LineFragments.back();

      Fragment.CodeSize = LF.header()->CodeSize;
      Fragment.Flags =
          static_cast<codeview::LineFlags>(uint16_t(LF.header()->Flags));
      Fragment.RelocOffset = LF.header()->RelocOffset;
      Fragment.RelocSegment = LF.header()->RelocSegment;

      for (const auto &L : LF) {
        Fragment.Blocks.emplace_back();
        auto &Block = Fragment.Blocks.back();

        if (auto Result = getNameFromChecksumsBuffer(L.NameIndex))
          Block.FileName = *Result;
        else
          return Result.takeError();

        for (const auto &N : L.LineNumbers) {
          CodeViewYAML::SourceLineEntry Line;
          Line.Offset = N.Offset;
          codeview::LineInfo LI(N.Flags);
          Line.LineStart = LI.getStartLine();
          Line.EndDelta = LI.getLineDelta();
          Line.IsStatement = LI.isStatement();
          Block.Lines.push_back(Line);
        }

        if (LF.hasColumnInfo()) {
          for (const auto &C : L.Columns) {
            CodeViewYAML::SourceColumnEntry Column;
            Column.StartColumn = C.StartColumn;
            Column.EndColumn = C.EndColumn;
            Block.Columns.push_back(Column);
          }
        }
      }
    }
    return Error::success();
  }

  Error handleInlineeLines() override {
    for (const auto &ILF : InlineeLines) {
      Info.Inlinees.emplace_back();
      auto &Inlinee = Info.Inlinees.back();

      Inlinee.HasExtraFiles = ILF.hasExtraFiles();
      for (const auto &IL : ILF) {
        Inlinee.Sites.emplace_back();
        auto &Site = Inlinee.Sites.back();
        if (auto Result = getNameFromChecksumsBuffer(IL.Header->FileID))
          Site.FileName = *Result;
        else
          return Result.takeError();

        Site.Inlinee = IL.Header->Inlinee.getIndex();
        Site.SourceLineNum = IL.Header->SourceLineNum;
        if (ILF.hasExtraFiles()) {
          for (const auto &EF : IL.ExtraFiles) {
            if (auto Result = getNameFromChecksumsBuffer(EF))
              Site.ExtraFiles.push_back(*Result);
            else
              return Result.takeError();
          }
        }
      }
    }
    return Error::success();
  }

private:
  CodeViewYAML::SourceFileInfo &Info;
};
}

Expected<Optional<CodeViewYAML::SourceFileInfo>>
YAMLOutputStyle::getFileLineInfo(const pdb::ModuleDebugStreamRef &ModS) {
  if (!ModS.hasLineInfo())
    return None;

  CodeViewYAML::SourceFileInfo Info;
  C13YamlVisitor Visitor(Info, File);
  if (auto EC =
          codeview::visitDebugSubsections(ModS.linesAndChecksums(), Visitor))
    return std::move(EC);

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
    auto S = ST.getStringForID(ID);
    if (!S)
      return S.takeError();
    if (S->empty())
      continue;
    Obj.StringTable->push_back(*S);
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
    const auto &Modules = DS.modules();
    for (uint32_t I = 0; I < Modules.getModuleCount(); ++I) {
      DbiModuleDescriptor MI = Modules.getModuleDescriptor(I);

      Obj.DbiStream->ModInfos.emplace_back();
      yaml::PdbDbiModuleInfo &DMI = Obj.DbiStream->ModInfos.back();

      DMI.Mod = MI.getModuleName();
      DMI.Obj = MI.getObjFileName();
      if (opts::pdb2yaml::DbiModuleSourceFileInfo) {
        auto Files = Modules.source_files(I);
        DMI.SourceFiles.assign(Files.begin(), Files.end());
      }

      uint16_t ModiStream = MI.getModuleStreamIndex();
      if (ModiStream == kInvalidStreamIndex)
        continue;

      auto ModStreamData = msf::MappedBlockStream::createIndexedStream(
          File.getMsfLayout(), File.getMsfBuffer(), ModiStream);

      pdb::ModuleDebugStreamRef ModS(MI, std::move(ModStreamData));
      if (auto EC = ModS.reload())
        return EC;

      if (opts::pdb2yaml::DbiModuleSourceLineInfo) {
        auto ExpectedInfo = getFileLineInfo(ModS);
        if (!ExpectedInfo)
          return ExpectedInfo.takeError();
        DMI.FileLineInfo = *ExpectedInfo;
      }

      if (opts::pdb2yaml::DbiModuleSyms) {
        DMI.Modi.emplace();

        DMI.Modi->Signature = ModS.signature();
        bool HadError = false;
        for (auto &Sym : ModS.symbols(&HadError)) {
          auto ES = CodeViewYAML::SymbolRecord::fromCodeViewSymbol(Sym);
          if (!ES)
            return ES.takeError();

          DMI.Modi->Symbols.push_back(*ES);
        }
      }
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
    auto ExpectedRecord = CodeViewYAML::LeafRecord::fromCodeViewRecord(Record);
    if (!ExpectedRecord)
      return ExpectedRecord.takeError();
    Obj.TpiStream->Records.push_back(*ExpectedRecord);
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
    auto ExpectedRecord = CodeViewYAML::LeafRecord::fromCodeViewRecord(Record);
    if (!ExpectedRecord)
      return ExpectedRecord.takeError();

    Obj.IpiStream->Records.push_back(*ExpectedRecord);
  }

  return Error::success();
}

void YAMLOutputStyle::flush() {
  Out << Obj;
  outs().flush();
}
