//===- DIASession.cpp - DIA implementation of IPDBSession -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumDebugStreams.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumLineNumbers.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumSourceFiles.h"
#include "llvm/DebugInfo/PDB/DIA/DIARawSymbol.h"
#include "llvm/DebugInfo/PDB/DIA/DIASession.h"
#include "llvm/DebugInfo/PDB/DIA/DIASourceFile.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/ConvertUTF.h"

#include <diacreate.h>

using namespace llvm;

namespace {

bool LoadDIA(CComPtr<IDiaDataSource>& DiaDataSource) {
  if (SUCCEEDED(CoCreateInstance(CLSID_DiaSource, nullptr, CLSCTX_INPROC_SERVER,
                                 IID_IDiaDataSource,
                                 reinterpret_cast<LPVOID *>(&DiaDataSource))))
    return true;

  // If the CoCreateInstance call above failed, msdia*.dll is not registered.
  // Try loading the DLL corresponding to the #included DIA SDK.
#if !defined(_MSC_VER)
  return false;
#else
  const wchar_t *msdia_dll = nullptr;
#if _MSC_VER == 1900
  msdia_dll = L"msdia140.dll"; // VS2015
#elif _MSC_VER == 1800
  msdia_dll = L"msdia120.dll"; // VS2013
#else
#error "Unknown Visual Studio version."
#endif
  return msdia_dll &&
         SUCCEEDED(NoRegCoCreate(msdia_dll, CLSID_DiaSource, IID_IDiaDataSource,
                                 reinterpret_cast<LPVOID *>(&DiaDataSource)));
#endif
}

}

DIASession::DIASession(CComPtr<IDiaSession> DiaSession) : Session(DiaSession) {}

PDB_ErrorCode DIASession::createFromPdb(StringRef Path,
                                        std::unique_ptr<IPDBSession> &Session) {
  CComPtr<IDiaDataSource> DiaDataSource;
  CComPtr<IDiaSession> DiaSession;

  // We assume that CoInitializeEx has already been called by the executable.
  if (!LoadDIA(DiaDataSource))
    return PDB_ErrorCode::NoPdbImpl;

  llvm::SmallVector<UTF16, 128> Path16;
  if (!llvm::convertUTF8ToUTF16String(Path, Path16))
    return PDB_ErrorCode::InvalidPath;

  const wchar_t *Path16Str = reinterpret_cast<const wchar_t*>(Path16.data());
  HRESULT Result;
  if (FAILED(Result = DiaDataSource->loadDataFromPdb(Path16Str))) {
    if (Result == E_PDB_NOT_FOUND)
      return PDB_ErrorCode::InvalidPath;
    else if (Result == E_PDB_FORMAT)
      return PDB_ErrorCode::InvalidFileFormat;
    else if (Result == E_INVALIDARG)
      return PDB_ErrorCode::InvalidParameter;
    else if (Result == E_UNEXPECTED)
      return PDB_ErrorCode::AlreadyLoaded;
    else
      return PDB_ErrorCode::UnknownError;
  }

  if (FAILED(Result = DiaDataSource->openSession(&DiaSession))) {
    if (Result == E_OUTOFMEMORY)
      return PDB_ErrorCode::NoMemory;
    else
      return PDB_ErrorCode::UnknownError;
  }

  Session.reset(new DIASession(DiaSession));
  return PDB_ErrorCode::Success;
}

PDB_ErrorCode DIASession::createFromExe(StringRef Path,
                                        std::unique_ptr<IPDBSession> &Session) {
  CComPtr<IDiaDataSource> DiaDataSource;
  CComPtr<IDiaSession> DiaSession;

  // We assume that CoInitializeEx has already been called by the executable.
  if (!LoadDIA(DiaDataSource))
    return PDB_ErrorCode::NoPdbImpl;

  llvm::SmallVector<UTF16, 128> Path16;
  if (!llvm::convertUTF8ToUTF16String(Path, Path16))
    return PDB_ErrorCode::InvalidPath;

  const wchar_t *Path16Str = reinterpret_cast<const wchar_t *>(Path16.data());
  HRESULT Result;
  if (FAILED(Result =
                 DiaDataSource->loadDataForExe(Path16Str, nullptr, nullptr))) {
    if (Result == E_PDB_NOT_FOUND)
      return PDB_ErrorCode::InvalidPath;
    else if (Result == E_PDB_FORMAT)
      return PDB_ErrorCode::InvalidFileFormat;
    else if (Result == E_PDB_INVALID_SIG || Result == E_PDB_INVALID_AGE)
      return PDB_ErrorCode::DebugInfoMismatch;
    else if (Result == E_INVALIDARG)
      return PDB_ErrorCode::InvalidParameter;
    else if (Result == E_UNEXPECTED)
      return PDB_ErrorCode::AlreadyLoaded;
    else
      return PDB_ErrorCode::UnknownError;
  }

  if (FAILED(Result = DiaDataSource->openSession(&DiaSession))) {
    if (Result == E_OUTOFMEMORY)
      return PDB_ErrorCode::NoMemory;
    else
      return PDB_ErrorCode::UnknownError;
  }

  Session.reset(new DIASession(DiaSession));
  return PDB_ErrorCode::Success;
}

uint64_t DIASession::getLoadAddress() const {
  uint64_t LoadAddress;
  bool success = (S_OK == Session->get_loadAddress(&LoadAddress));
  return (success) ? LoadAddress : 0;
}

void DIASession::setLoadAddress(uint64_t Address) {
  Session->put_loadAddress(Address);
}

std::unique_ptr<PDBSymbolExe> DIASession::getGlobalScope() const {
  CComPtr<IDiaSymbol> GlobalScope;
  if (S_OK != Session->get_globalScope(&GlobalScope))
    return nullptr;

  auto RawSymbol = llvm::make_unique<DIARawSymbol>(*this, GlobalScope);
  auto PdbSymbol(PDBSymbol::create(*this, std::move(RawSymbol)));
  std::unique_ptr<PDBSymbolExe> ExeSymbol(
      static_cast<PDBSymbolExe *>(PdbSymbol.release()));
  return ExeSymbol;
}

std::unique_ptr<PDBSymbol> DIASession::getSymbolById(uint32_t SymbolId) const {
  CComPtr<IDiaSymbol> LocatedSymbol;
  if (S_OK != Session->symbolById(SymbolId, &LocatedSymbol))
    return nullptr;

  auto RawSymbol = llvm::make_unique<DIARawSymbol>(*this, LocatedSymbol);
  return PDBSymbol::create(*this, std::move(RawSymbol));
}

std::unique_ptr<PDBSymbol>
DIASession::findSymbolByAddress(uint64_t Address, PDB_SymType Type) const {
  enum SymTagEnum EnumVal = static_cast<enum SymTagEnum>(Type);

  CComPtr<IDiaSymbol> Symbol;
  if (S_OK != Session->findSymbolByVA(Address, EnumVal, &Symbol)) {
    ULONGLONG LoadAddr = 0;
    if (S_OK != Session->get_loadAddress(&LoadAddr))
      return nullptr;
    DWORD RVA = static_cast<DWORD>(Address - LoadAddr);
    if (S_OK != Session->findSymbolByRVA(RVA, EnumVal, &Symbol))
      return nullptr;
  }
  auto RawSymbol = llvm::make_unique<DIARawSymbol>(*this, Symbol);
  return PDBSymbol::create(*this, std::move(RawSymbol));
}

std::unique_ptr<IPDBEnumLineNumbers>
DIASession::findLineNumbers(const PDBSymbolCompiland &Compiland,
                            const IPDBSourceFile &File) const {
  const DIARawSymbol &RawCompiland =
      static_cast<const DIARawSymbol &>(Compiland.getRawSymbol());
  const DIASourceFile &RawFile = static_cast<const DIASourceFile &>(File);

  CComPtr<IDiaEnumLineNumbers> LineNumbers;
  if (S_OK !=
      Session->findLines(RawCompiland.getDiaSymbol(), RawFile.getDiaFile(),
                         &LineNumbers))
    return nullptr;

  return llvm::make_unique<DIAEnumLineNumbers>(LineNumbers);
}

std::unique_ptr<IPDBEnumLineNumbers>
DIASession::findLineNumbersByAddress(uint64_t Address, uint32_t Length) const {
  CComPtr<IDiaEnumLineNumbers> LineNumbers;
  if (S_OK != Session->findLinesByVA(Address, Length, &LineNumbers))
    return nullptr;

  return llvm::make_unique<DIAEnumLineNumbers>(LineNumbers);
}

std::unique_ptr<IPDBEnumSourceFiles>
DIASession::findSourceFiles(const PDBSymbolCompiland *Compiland,
                            llvm::StringRef Pattern,
                            PDB_NameSearchFlags Flags) const {
  IDiaSymbol *DiaCompiland = nullptr;
  CComBSTR Utf16Pattern;
  if (!Pattern.empty())
    Utf16Pattern = CComBSTR(Pattern.data());

  if (Compiland)
    DiaCompiland = static_cast<const DIARawSymbol &>(Compiland->getRawSymbol())
                       .getDiaSymbol();

  Flags = static_cast<PDB_NameSearchFlags>(
      Flags | PDB_NameSearchFlags::NS_FileNameExtMatch);
  CComPtr<IDiaEnumSourceFiles> SourceFiles;
  if (S_OK !=
      Session->findFile(DiaCompiland, Utf16Pattern.m_str, Flags, &SourceFiles))
    return nullptr;
  return llvm::make_unique<DIAEnumSourceFiles>(*this, SourceFiles);
}

std::unique_ptr<IPDBSourceFile>
DIASession::findOneSourceFile(const PDBSymbolCompiland *Compiland,
                              llvm::StringRef Pattern,
                              PDB_NameSearchFlags Flags) const {
  auto SourceFiles = findSourceFiles(Compiland, Pattern, Flags);
  if (!SourceFiles || SourceFiles->getChildCount() == 0)
    return nullptr;
  return SourceFiles->getNext();
}

std::unique_ptr<IPDBEnumChildren<PDBSymbolCompiland>>
DIASession::findCompilandsForSourceFile(llvm::StringRef Pattern,
                                        PDB_NameSearchFlags Flags) const {
  auto File = findOneSourceFile(nullptr, Pattern, Flags);
  if (!File)
    return nullptr;
  return File->getCompilands();
}

std::unique_ptr<PDBSymbolCompiland>
DIASession::findOneCompilandForSourceFile(llvm::StringRef Pattern,
                                          PDB_NameSearchFlags Flags) const {
  auto Compilands = findCompilandsForSourceFile(Pattern, Flags);
  if (!Compilands || Compilands->getChildCount() == 0)
    return nullptr;
  return Compilands->getNext();
}

std::unique_ptr<IPDBEnumSourceFiles> DIASession::getAllSourceFiles() const {
  CComPtr<IDiaEnumSourceFiles> Files;
  if (S_OK != Session->findFile(nullptr, nullptr, nsNone, &Files))
    return nullptr;

  return llvm::make_unique<DIAEnumSourceFiles>(*this, Files);
}

std::unique_ptr<IPDBEnumSourceFiles> DIASession::getSourceFilesForCompiland(
    const PDBSymbolCompiland &Compiland) const {
  CComPtr<IDiaEnumSourceFiles> Files;

  const DIARawSymbol &RawSymbol =
      static_cast<const DIARawSymbol &>(Compiland.getRawSymbol());
  if (S_OK !=
      Session->findFile(RawSymbol.getDiaSymbol(), nullptr, nsNone, &Files))
    return nullptr;

  return llvm::make_unique<DIAEnumSourceFiles>(*this, Files);
}

std::unique_ptr<IPDBSourceFile>
DIASession::getSourceFileById(uint32_t FileId) const {
  CComPtr<IDiaSourceFile> LocatedFile;
  if (S_OK != Session->findFileById(FileId, &LocatedFile))
    return nullptr;

  return llvm::make_unique<DIASourceFile>(*this, LocatedFile);
}

std::unique_ptr<IPDBEnumDataStreams> DIASession::getDebugStreams() const {
  CComPtr<IDiaEnumDebugStreams> DiaEnumerator;
  if (S_OK != Session->getEnumDebugStreams(&DiaEnumerator))
    return nullptr;

  return llvm::make_unique<DIAEnumDebugStreams>(DiaEnumerator);
}
