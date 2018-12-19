//===- Reader.cpp ---------------------------------------------------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Reader.h"
#include "Object.h"
#include "llvm-objcopy.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>

namespace llvm {
namespace objcopy {
namespace coff {

using namespace object;

Reader::~Reader() {}

void COFFReader::readExecutableHeaders(Object &Obj) const {
  const dos_header *DH = COFFObj.getDOSHeader();
  Obj.Is64 = COFFObj.is64();
  if (!DH)
    return;

  Obj.IsPE = true;
  Obj.DosHeader = *DH;
  if (DH->AddressOfNewExeHeader > sizeof(*DH))
    Obj.DosStub = ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&DH[1]),
                                    DH->AddressOfNewExeHeader - sizeof(*DH));

  if (COFFObj.is64()) {
    const pe32plus_header *PE32Plus = nullptr;
    if (auto EC = COFFObj.getPE32PlusHeader(PE32Plus))
      reportError(COFFObj.getFileName(), std::move(EC));
    Obj.PeHeader = *PE32Plus;
  } else {
    const pe32_header *PE32 = nullptr;
    if (auto EC = COFFObj.getPE32Header(PE32))
      reportError(COFFObj.getFileName(), std::move(EC));
    copyPeHeader(Obj.PeHeader, *PE32);
    // The pe32plus_header (stored in Object) lacks the BaseOfData field.
    Obj.BaseOfData = PE32->BaseOfData;
  }

  for (size_t I = 0; I < Obj.PeHeader.NumberOfRvaAndSize; I++) {
    const data_directory *Dir;
    if (auto EC = COFFObj.getDataDirectory(I, Dir))
      reportError(COFFObj.getFileName(), std::move(EC));
    Obj.DataDirectories.emplace_back(*Dir);
  }
}

void COFFReader::readSections(Object &Obj) const {
  // Section indexing starts from 1.
  for (size_t I = 1, E = COFFObj.getNumberOfSections(); I <= E; I++) {
    const coff_section *Sec;
    if (auto EC = COFFObj.getSection(I, Sec))
      reportError(COFFObj.getFileName(), std::move(EC));
    Obj.Sections.push_back(Section());
    Section &S = Obj.Sections.back();
    S.Header = *Sec;
    if (auto EC = COFFObj.getSectionContents(Sec, S.Contents))
      reportError(COFFObj.getFileName(), std::move(EC));
    ArrayRef<coff_relocation> Relocs = COFFObj.getRelocations(Sec);
    S.Relocs.insert(S.Relocs.end(), Relocs.begin(), Relocs.end());
    if (auto EC = COFFObj.getSectionName(Sec, S.Name))
      reportError(COFFObj.getFileName(), std::move(EC));
    if (Sec->hasExtendedRelocations())
      reportError(
          COFFObj.getFileName(),
          make_error<StringError>("Extended relocations not supported yet",
                                  object_error::parse_failed));
  }
}

void COFFReader::readSymbols(Object &Obj, bool IsBigObj) const {
  for (uint32_t I = 0, E = COFFObj.getRawNumberOfSymbols(); I < E;) {
    Expected<COFFSymbolRef> SymOrErr = COFFObj.getSymbol(I);
    if (!SymOrErr)
      reportError(COFFObj.getFileName(), SymOrErr.takeError());
    COFFSymbolRef SymRef = *SymOrErr;

    Obj.Symbols.push_back(Symbol());
    Symbol &Sym = Obj.Symbols.back();
    // Copy symbols from the original form into an intermediate coff_symbol32.
    if (IsBigObj)
      copySymbol(Sym.Sym,
                 *reinterpret_cast<const coff_symbol32 *>(SymRef.getRawPtr()));
    else
      copySymbol(Sym.Sym,
                 *reinterpret_cast<const coff_symbol16 *>(SymRef.getRawPtr()));
    if (auto EC = COFFObj.getSymbolName(SymRef, Sym.Name))
      reportError(COFFObj.getFileName(), std::move(EC));
    Sym.AuxData = COFFObj.getSymbolAuxData(SymRef);
    assert((Sym.AuxData.size() %
            (IsBigObj ? sizeof(coff_symbol32) : sizeof(coff_symbol16))) == 0);
    I += 1 + SymRef.getNumberOfAuxSymbols();
  }
}

std::unique_ptr<Object> COFFReader::create() const {
  auto Obj = llvm::make_unique<Object>();

  const coff_file_header *CFH = nullptr;
  const coff_bigobj_file_header *CBFH = nullptr;
  COFFObj.getCOFFHeader(CFH);
  COFFObj.getCOFFBigObjHeader(CBFH);
  bool IsBigObj = false;
  if (CFH) {
    Obj->CoffFileHeader = *CFH;
  } else {
    if (!CBFH)
      reportError(COFFObj.getFileName(),
                  make_error<StringError>("No COFF file header returned",
                                          object_error::parse_failed));
    // Only copying the few fields from the bigobj header that we need
    // and won't recreate in the end.
    Obj->CoffFileHeader.Machine = CBFH->Machine;
    Obj->CoffFileHeader.TimeDateStamp = CBFH->TimeDateStamp;
    IsBigObj = true;
  }

  readExecutableHeaders(*Obj);
  readSections(*Obj);
  readSymbols(*Obj, IsBigObj);

  return Obj;
}

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm
