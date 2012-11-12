//=-- llvm/CodeGen/DwarfAccelTable.cpp - Dwarf Accelerator Tables -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf accelerator tables.
//
//===----------------------------------------------------------------------===//

#include "DwarfAccelTable.h"
#include "DwarfDebug.h"
#include "DIE.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

const char *DwarfAccelTable::Atom::AtomTypeString(enum AtomType AT) {
  switch (AT) {
  case eAtomTypeNULL: return "eAtomTypeNULL";
  case eAtomTypeDIEOffset: return "eAtomTypeDIEOffset";
  case eAtomTypeCUOffset: return "eAtomTypeCUOffset";
  case eAtomTypeTag: return "eAtomTypeTag";
  case eAtomTypeNameFlags: return "eAtomTypeNameFlags";
  case eAtomTypeTypeFlags: return "eAtomTypeTypeFlags";
  } 
  llvm_unreachable("invalid AtomType!");
}

// The length of the header data is always going to be 4 + 4 + 4*NumAtoms.
DwarfAccelTable::DwarfAccelTable(ArrayRef<DwarfAccelTable::Atom> atomList) :
  Header(8 + (atomList.size() * 4)),
  HeaderData(atomList),
  Entries(Allocator) { }

DwarfAccelTable::~DwarfAccelTable() { }

void DwarfAccelTable::AddName(StringRef Name, DIE* die, char Flags) {
  assert(Data.empty() && "Already finalized!");
  // If the string is in the list already then add this die to the list
  // otherwise add a new one.
  DataArray &DIEs = Entries[Name];
  DIEs.push_back(new (Allocator) HashDataContents(die, Flags));
}

void DwarfAccelTable::ComputeBucketCount(void) {
  // First get the number of unique hashes.
  std::vector<uint32_t> uniques(Data.size());
  for (size_t i = 0, e = Data.size(); i < e; ++i)
    uniques[i] = Data[i]->HashValue;
  array_pod_sort(uniques.begin(), uniques.end());
  std::vector<uint32_t>::iterator p =
    std::unique(uniques.begin(), uniques.end());
  uint32_t num = std::distance(uniques.begin(), p);

  // Then compute the bucket size, minimum of 1 bucket.
  if (num > 1024) Header.bucket_count = num/4;
  if (num > 16) Header.bucket_count = num/2;
  else Header.bucket_count = num > 0 ? num : 1;

  Header.hashes_count = num;
}

// compareDIEs - comparison predicate that sorts DIEs by their offset.
static bool compareDIEs(const DwarfAccelTable::HashDataContents *A,
                        const DwarfAccelTable::HashDataContents *B) {
  return A->Die->getOffset() < B->Die->getOffset();
}

void DwarfAccelTable::FinalizeTable(AsmPrinter *Asm, const char *Prefix) {
  // Create the individual hash data outputs.
  for (StringMap<DataArray>::iterator
         EI = Entries.begin(), EE = Entries.end(); EI != EE; ++EI) {

    // Unique the entries.
    std::stable_sort(EI->second.begin(), EI->second.end(), compareDIEs);
    EI->second.erase(std::unique(EI->second.begin(), EI->second.end()),
                       EI->second.end());

    HashData *Entry = new (Allocator) HashData(EI->getKey(), EI->second);
    Data.push_back(Entry);
  }

  // Figure out how many buckets we need, then compute the bucket
  // contents and the final ordering. We'll emit the hashes and offsets
  // by doing a walk during the emission phase. We add temporary
  // symbols to the data so that we can reference them during the offset
  // later, we'll emit them when we emit the data.
  ComputeBucketCount();

  // Compute bucket contents and final ordering.
  Buckets.resize(Header.bucket_count);
  for (size_t i = 0, e = Data.size(); i < e; ++i) {
    uint32_t bucket = Data[i]->HashValue % Header.bucket_count;
    Buckets[bucket].push_back(Data[i]);
    Data[i]->Sym = Asm->GetTempSymbol(Prefix, i);
  }
}

// Emits the header for the table via the AsmPrinter.
void DwarfAccelTable::EmitHeader(AsmPrinter *Asm) {
  Asm->OutStreamer.AddComment("Header Magic");
  Asm->EmitInt32(Header.magic);
  Asm->OutStreamer.AddComment("Header Version");
  Asm->EmitInt16(Header.version);
  Asm->OutStreamer.AddComment("Header Hash Function");
  Asm->EmitInt16(Header.hash_function);
  Asm->OutStreamer.AddComment("Header Bucket Count");
  Asm->EmitInt32(Header.bucket_count);
  Asm->OutStreamer.AddComment("Header Hash Count");
  Asm->EmitInt32(Header.hashes_count);
  Asm->OutStreamer.AddComment("Header Data Length");
  Asm->EmitInt32(Header.header_data_len);
  Asm->OutStreamer.AddComment("HeaderData Die Offset Base");
  Asm->EmitInt32(HeaderData.die_offset_base);
  Asm->OutStreamer.AddComment("HeaderData Atom Count");
  Asm->EmitInt32(HeaderData.Atoms.size());
  for (size_t i = 0; i < HeaderData.Atoms.size(); i++) {
    Atom A = HeaderData.Atoms[i];
    Asm->OutStreamer.AddComment(Atom::AtomTypeString(A.type));
    Asm->EmitInt16(A.type);
    Asm->OutStreamer.AddComment(dwarf::FormEncodingString(A.form));
    Asm->EmitInt16(A.form);
  }
}

// Walk through and emit the buckets for the table. Each index is
// an offset into the list of hashes.
void DwarfAccelTable::EmitBuckets(AsmPrinter *Asm) {
  unsigned index = 0;
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    Asm->OutStreamer.AddComment("Bucket " + Twine(i));
    if (Buckets[i].size() != 0)
      Asm->EmitInt32(index);
    else
      Asm->EmitInt32(UINT32_MAX);
    index += Buckets[i].size();
  }
}

// Walk through the buckets and emit the individual hashes for each
// bucket.
void DwarfAccelTable::EmitHashes(AsmPrinter *Asm) {
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    for (HashList::const_iterator HI = Buckets[i].begin(),
           HE = Buckets[i].end(); HI != HE; ++HI) {
      Asm->OutStreamer.AddComment("Hash in Bucket " + Twine(i));
      Asm->EmitInt32((*HI)->HashValue);
    } 
  }
}

// Walk through the buckets and emit the individual offsets for each
// element in each bucket. This is done via a symbol subtraction from the
// beginning of the section. The non-section symbol will be output later
// when we emit the actual data.
void DwarfAccelTable::EmitOffsets(AsmPrinter *Asm, MCSymbol *SecBegin) {
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    for (HashList::const_iterator HI = Buckets[i].begin(),
           HE = Buckets[i].end(); HI != HE; ++HI) {
      Asm->OutStreamer.AddComment("Offset in Bucket " + Twine(i));
      MCContext &Context = Asm->OutStreamer.getContext();
      const MCExpr *Sub =
        MCBinaryExpr::CreateSub(MCSymbolRefExpr::Create((*HI)->Sym, Context),
                                MCSymbolRefExpr::Create(SecBegin, Context),
                                Context);
      Asm->OutStreamer.EmitValue(Sub, sizeof(uint32_t), 0);
    }
  }
}

// Walk through the buckets and emit the full data for each element in
// the bucket. For the string case emit the dies and the various offsets.
// Terminate each HashData bucket with 0.
void DwarfAccelTable::EmitData(AsmPrinter *Asm, DwarfDebug *D) {
  uint64_t PrevHash = UINT64_MAX;
  for (size_t i = 0, e = Buckets.size(); i < e; ++i) {
    for (HashList::const_iterator HI = Buckets[i].begin(),
           HE = Buckets[i].end(); HI != HE; ++HI) {
      // Remember to emit the label for our offset.
      Asm->OutStreamer.EmitLabel((*HI)->Sym);
      Asm->OutStreamer.AddComment((*HI)->Str);
      Asm->EmitSectionOffset(D->getStringPoolEntry((*HI)->Str),
                             D->getStringPool());
      Asm->OutStreamer.AddComment("Num DIEs");
      Asm->EmitInt32((*HI)->Data.size());
      for (ArrayRef<HashDataContents*>::const_iterator
             DI = (*HI)->Data.begin(), DE = (*HI)->Data.end();
           DI != DE; ++DI) {
        // Emit the DIE offset
        Asm->EmitInt32((*DI)->Die->getOffset());
        // If we have multiple Atoms emit that info too.
        // FIXME: A bit of a hack, we either emit only one atom or all info.
        if (HeaderData.Atoms.size() > 1) {
          Asm->EmitInt16((*DI)->Die->getTag());
          Asm->EmitInt8((*DI)->Flags);
        }
      }
      // Emit a 0 to terminate the data unless we have a hash collision.
      if (PrevHash != (*HI)->HashValue)
        Asm->EmitInt32(0);
      PrevHash = (*HI)->HashValue;
    }
  }
}

// Emit the entire data structure to the output file.
void DwarfAccelTable::Emit(AsmPrinter *Asm, MCSymbol *SecBegin,
                           DwarfDebug *D) {
  // Emit the header.
  EmitHeader(Asm);

  // Emit the buckets.
  EmitBuckets(Asm);

  // Emit the hashes.
  EmitHashes(Asm);

  // Emit the offsets.
  EmitOffsets(Asm, SecBegin);

  // Emit the hash data.
  EmitData(Asm, D);
}

#ifndef NDEBUG
void DwarfAccelTable::print(raw_ostream &O) {

  Header.print(O);
  HeaderData.print(O);

  O << "Entries: \n";
  for (StringMap<DataArray>::const_iterator
         EI = Entries.begin(), EE = Entries.end(); EI != EE; ++EI) {
    O << "Name: " << EI->getKeyData() << "\n";
    for (DataArray::const_iterator DI = EI->second.begin(),
           DE = EI->second.end();
         DI != DE; ++DI)
      (*DI)->print(O);
  }

  O << "Buckets and Hashes: \n";
  for (size_t i = 0, e = Buckets.size(); i < e; ++i)
    for (HashList::const_iterator HI = Buckets[i].begin(),
           HE = Buckets[i].end(); HI != HE; ++HI)
      (*HI)->print(O);

  O << "Data: \n";
    for (std::vector<HashData*>::const_iterator
           DI = Data.begin(), DE = Data.end(); DI != DE; ++DI)
      (*DI)->print(O);
  

}
#endif
