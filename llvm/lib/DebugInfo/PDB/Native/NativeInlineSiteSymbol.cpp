//===- NativeInlineSiteSymbol.cpp - info about inline sites -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeInlineSiteSymbol.h"

#include "llvm/DebugInfo/CodeView/DebugInlineeLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeEnumLineNumbers.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NativeInlineSiteSymbol::NativeInlineSiteSymbol(
    NativeSession &Session, SymIndexId Id, const codeview::InlineSiteSym &Sym,
    uint64_t ParentAddr)
    : NativeRawSymbol(Session, PDB_SymType::InlineSite, Id), Sym(Sym),
      ParentAddr(ParentAddr) {}

NativeInlineSiteSymbol::~NativeInlineSiteSymbol() = default;

void NativeInlineSiteSymbol::dump(raw_ostream &OS, int Indent,
                                  PdbSymbolIdField ShowIdFields,
                                  PdbSymbolIdField RecurseIdFields) const {
  NativeRawSymbol::dump(OS, Indent, ShowIdFields, RecurseIdFields);
  dumpSymbolField(OS, "name", getName(), Indent);
}

static Optional<InlineeSourceLine>
findInlineeByTypeIndex(TypeIndex Id, ModuleDebugStreamRef &ModS) {
  for (const auto &SS : ModS.getSubsectionsArray()) {
    if (SS.kind() != DebugSubsectionKind::InlineeLines)
      continue;

    DebugInlineeLinesSubsectionRef InlineeLines;
    BinaryStreamReader Reader(SS.getRecordData());
    if (auto EC = InlineeLines.initialize(Reader)) {
      consumeError(std::move(EC));
      continue;
    }

    for (const InlineeSourceLine &Line : InlineeLines)
      if (Line.Header->Inlinee == Id)
        return Line;
  }
  return None;
}

std::string NativeInlineSiteSymbol::getName() const {
  auto Tpi = Session.getPDBFile().getPDBTpiStream();
  if (!Tpi) {
    consumeError(Tpi.takeError());
    return "";
  }
  auto Ipi = Session.getPDBFile().getPDBIpiStream();
  if (!Ipi) {
    consumeError(Ipi.takeError());
    return "";
  }

  LazyRandomTypeCollection &Types = Tpi->typeCollection();
  LazyRandomTypeCollection &Ids = Ipi->typeCollection();
  CVType InlineeType = Ids.getType(Sym.Inlinee);
  std::string QualifiedName;
  if (InlineeType.kind() == LF_MFUNC_ID) {
    MemberFuncIdRecord MFRecord;
    cantFail(TypeDeserializer::deserializeAs<MemberFuncIdRecord>(InlineeType,
                                                                 MFRecord));
    TypeIndex ClassTy = MFRecord.getClassType();
    QualifiedName.append(std::string(Types.getTypeName(ClassTy)));
    QualifiedName.append("::");
  } else if (InlineeType.kind() == LF_FUNC_ID) {
    FuncIdRecord FRecord;
    cantFail(
        TypeDeserializer::deserializeAs<FuncIdRecord>(InlineeType, FRecord));
    TypeIndex ParentScope = FRecord.getParentScope();
    if (!ParentScope.isNoneType()) {
      QualifiedName.append(std::string(Ids.getTypeName(ParentScope)));
      QualifiedName.append("::");
    }
  }

  QualifiedName.append(std::string(Ids.getTypeName(Sym.Inlinee)));
  return QualifiedName;
}

void NativeInlineSiteSymbol::getLineOffset(uint32_t OffsetInFunc,
                                           uint32_t &LineOffset,
                                           uint32_t &FileOffset) const {
  LineOffset = 0;
  FileOffset = 0;
  uint32_t CodeOffset = 0;
  for (const auto &Annot : Sym.annotations()) {
    switch (Annot.OpCode) {
    case BinaryAnnotationsOpCode::CodeOffset:
    case BinaryAnnotationsOpCode::ChangeCodeOffset:
    case BinaryAnnotationsOpCode::ChangeCodeLength:
      CodeOffset += Annot.U1;
      break;
    case BinaryAnnotationsOpCode::ChangeCodeLengthAndCodeOffset:
      CodeOffset += Annot.U2;
      break;
    case BinaryAnnotationsOpCode::ChangeLineOffset:
    case BinaryAnnotationsOpCode::ChangeCodeOffsetAndLineOffset:
      CodeOffset += Annot.U1;
      LineOffset += Annot.S1;
      break;
    case BinaryAnnotationsOpCode::ChangeFile:
      FileOffset = Annot.U1;
      break;
    default:
      break;
    }

    if (CodeOffset >= OffsetInFunc)
      return;
  }
}

std::unique_ptr<IPDBEnumLineNumbers>
NativeInlineSiteSymbol::findInlineeLinesByVA(uint64_t VA,
                                             uint32_t Length) const {
  uint16_t Modi;
  if (!Session.moduleIndexForVA(VA, Modi))
    return nullptr;

  Expected<ModuleDebugStreamRef> ModS = Session.getModuleDebugStream(Modi);
  if (!ModS) {
    consumeError(ModS.takeError());
    return nullptr;
  }

  Expected<DebugChecksumsSubsectionRef> Checksums =
      ModS->findChecksumsSubsection();
  if (!Checksums) {
    consumeError(Checksums.takeError());
    return nullptr;
  }

  // Get the line number offset and source file offset.
  uint32_t SrcLineOffset;
  uint32_t SrcFileOffset;
  getLineOffset(VA - ParentAddr, SrcLineOffset, SrcFileOffset);

  // Get line info from inlinee line table.
  Optional<InlineeSourceLine> Inlinee =
      findInlineeByTypeIndex(Sym.Inlinee, ModS.get());

  if (!Inlinee)
    return nullptr;

  uint32_t SrcLine = Inlinee->Header->SourceLineNum + SrcLineOffset;
  uint32_t SrcCol = 0; // Inline sites don't seem to have column info.
  uint32_t FileChecksumOffset =
      (SrcFileOffset == 0) ? Inlinee->Header->FileID : SrcFileOffset;

  auto ChecksumIter = Checksums->getArray().at(FileChecksumOffset);
  uint32_t SrcFileId =
      Session.getSymbolCache().getOrCreateSourceFile(*ChecksumIter);

  uint32_t LineSect, LineOff;
  Session.addressForVA(VA, LineSect, LineOff);
  NativeLineNumber LineNum(Session, SrcLine, SrcCol, LineSect, LineOff, Length,
                           SrcFileId, Modi);
  auto SrcFile = Session.getSymbolCache().getSourceFileById(SrcFileId);
  std::vector<NativeLineNumber> Lines{LineNum};

  return std::make_unique<NativeEnumLineNumbers>(std::move(Lines));
}
