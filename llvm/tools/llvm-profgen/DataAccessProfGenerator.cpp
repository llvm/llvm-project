//===- DataAccessProfGenerator.cpp - Data access profile ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DataAccessProfGenerator.h"
#include "PerfReader.h"
#include "ProfiledBinary.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/WithColor.h"
#include <map>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::sampleprof;
using memprof::SymbolHandleRef;

namespace {

// A data object from the symbol table, covering [Start, End) at its preferred
// ELF VA.
struct DataSymbol {
  uint64_t Start = 0;
  uint64_t End = 0;
  // The profile key: the canonical name, or for a string literal the MD5 of
  // its bytes. Names point into the binary, which outlives the profile.
  SymbolHandleRef Key;
  // A default-visibility global that another loaded file also defines. That
  // copy may take all of its accesses, so no samples does not make it cold.
  bool Preemptible = false;
};

} // namespace

static bool isStringLiteralName(StringRef Name) {
  // MemProfUse looks up IR names that start with `.str` by content hash.
  // Clang's ELF symbol for those literals is often `.L.str` / `.L.str.N`.
  return Name.starts_with(".str") || Name.starts_with(".L.str");
}

// Collect the data objects of \p Obj, sorted by address. These are the only
// objects the profile can name, hot or cold.
static Error
collectDataSymbols(const ELFObjectFileBase &Obj,
                   function_ref<bool(StringRef)> IsDefinedElsewhere,
                   std::vector<DataSymbol> &Symbols) {
  for (ELFSymbolRef Sym : Obj.symbols()) {
    // Data objects are STT_OBJECT or STT_COMMON. Assembly and linker-defined
    // data is often STT_NOTYPE. A zero-sized symbol is a label and cannot
    // contain a sampled address.
    uint64_t Size = Sym.getSize();
    uint8_t Type = Sym.getELFType();
    if (Size == 0 || (Type != ELF::STT_OBJECT && Type != ELF::STT_COMMON &&
                      Type != ELF::STT_NOTYPE))
      continue;

    Expected<uint32_t> FlagsOrErr = Sym.getFlags();
    if (!FlagsOrErr)
      return FlagsOrErr.takeError();
    // Same symbols llvm-objdump ignores: no definition in this file, an
    // absolute constant, or a format-specific symbol (null, file, section,
    // mapping).
    if (*FlagsOrErr & (SymbolRef::SF_Undefined | SymbolRef::SF_Absolute |
                       SymbolRef::SF_FormatSpecific))
      continue;

    // SHN_COMMON symbols have no section. isData() is allocated
    // non-executable file data; isBSS() is SHT_NOBITS.
    std::optional<SectionRef> Sec;
    if (!(*FlagsOrErr & SymbolRef::SF_Common)) {
      Expected<section_iterator> SecOrErr = Sym.getSection();
      if (!SecOrErr)
        return SecOrErr.takeError();
      if (*SecOrErr == Obj.section_end() ||
          !((*SecOrErr)->isData() || (*SecOrErr)->isBSS()))
        continue;
      Sec = **SecOrErr;
    }

    Expected<StringRef> NameOrErr = Sym.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();
    // The profile has no key for a nameless object.
    if (NameOrErr->empty())
      continue;
    Expected<uint64_t> AddrOrErr = Sym.getAddress();
    if (!AddrOrErr)
      return AddrOrErr.takeError();

    DataSymbol DS;
    uint64_t Start = *AddrOrErr;
    // A wrapped end covers almost the whole address space and would mark the
    // symbol extremely hot.
    if (Start + Size < Start)
      continue;
    DS.Start = Start;
    DS.End = Start + Size;
    if (!isStringLiteralName(*NameOrErr)) {
      // setDataAccessProfile canonicalizes names. Aggregate under that key so
      // foo.llvm.1 and foo.llvm.2 merge instead of aborting as duplicates, and
      // so an unsampled spelling is not also recorded as known-cold.
      StringRef Canonical = InstrProfSymtab::getCanonicalName(*NameOrErr);
      DS.Key = Canonical;
      // Probe the canonical key. foo.llvm.1 in this file and foo in another
      // are the same object.
      DS.Preemptible = Sym.getBinding() != ELF::STB_LOCAL &&
                       (Sym.getOther() & 0x3) == ELF::STV_DEFAULT &&
                       IsDefinedElsewhere(Canonical);
      Symbols.push_back(DS);
      continue;
    }

    // A string literal is keyed by the MD5 of its bytes, as MemProfUse does
    // for `.str` globals, so its bytes must be in the file. substr clamps, so
    // a literal that does not fit in its section comes back short.
    StringRef Bytes;
    if (Sec && !Sec->isBSS()) {
      Expected<StringRef> ContentsOrErr = Sec->getContents();
      if (!ContentsOrErr)
        return ContentsOrErr.takeError();
      if (DS.Start >= Sec->getAddress())
        Bytes = ContentsOrErr->substr(DS.Start - Sec->getAddress(), Size);
    }
    if (Bytes.size() != Size) {
      WithColor::warning() << "Cannot read contents of string literal "
                           << *NameOrErr << " in " << Obj.getFileName()
                           << ", skipping\n";
      continue;
    }
    DS.Key = MD5Hash(Bytes);
    Symbols.push_back(DS);
  }
  // Sorted by address, with the key breaking ties, so the known-cold order in
  // the output does not depend on symbol table order.
  llvm::sort(Symbols, [](const DataSymbol &A, const DataSymbol &B) {
    return std::tie(A.Start, A.End, A.Key) < std::tie(B.Start, B.End, B.Key);
  });
  return Error::success();
}

// Fill \p DAP from the sorted sampled addresses \p Addrs, one entry per
// sample. Each sample is credited to every symbol that covers it, so aliases
// and tail-merged literals all get the shared samples. Symbols with no
// samples become known-cold.
static Error buildDataAccessProf(ArrayRef<DataSymbol> Symbols,
                                 ArrayRef<uint64_t> Addrs, bool AddKnownCold,
                                 memprof::DataAccessProfData &DAP) {
  // Several symbols can have one key, e.g. foo.llvm.1 and foo.llvm.2, or
  // identical literals from different modules. setDataAccessProfile rejects a
  // key that is added twice, so sum per key first.
  std::map<SymbolHandleRef, uint64_t> Counts;
  std::map<SymbolHandleRef, bool> Preemptible;
  for (const DataSymbol &S : Symbols) {
    Preemptible[S.Key] |= S.Preemptible;
    // The number of samples in [Start, End).
    uint64_t Count =
        llvm::lower_bound(Addrs, S.End) - llvm::lower_bound(Addrs, S.Start);
    if (Count)
      Counts[S.Key] += Count;
  }
  for (const auto &[Key, Count] : Counts)
    if (Error E = DAP.setDataAccessProfile(Key, Count))
      return E;
  if (!AddKnownCold)
    return Error::success();

  // A data object with no samples is known-cold, so the compiler can move it
  // to an .unlikely section. A key that was sampled under any spelling is not
  // known-cold. The known-cold sets drop duplicates.
  for (const DataSymbol &S : Symbols) {
    if (Preemptible[S.Key] || DAP.getProfileRecord(S.Key))
      continue;
    if (Error E = DAP.addKnownSymbolWithoutSamples(S.Key))
      return E;
  }
  return Error::success();
}

// Names of the data objects that \p Files define in their dynamic symbol
// tables. A shared library's variable of the same name may get none of its
// accesses: an executable that uses it gets a copy relocation, and the
// executable or an earlier library may define it. Returns std::nullopt if a
// file cannot be read.
static std::optional<StringSet<>>
getDynamicDataObjects(const StringSet<> &Files) {
  StringSet<> Names;
  for (const auto &File : Files) {
    StringRef Path = File.getKey();
    // Files that are not ELF, e.g. JIT caches, are skipped rather than
    // treated as unreadable.
    file_magic Magic;
    if (std::error_code EC = identify_magic(Path, Magic)) {
      WithColor::warning() << "cannot read " << Path << ": " << EC.message()
                           << "\n";
      return std::nullopt;
    }
    if (Magic != file_magic::elf_executable &&
        Magic != file_magic::elf_shared_object)
      continue;
    Expected<OwningBinary<Binary>> BinOrErr = createBinary(Path);
    if (!BinOrErr) {
      WithColor::warning() << "cannot read " << Path << ": "
                           << toString(BinOrErr.takeError()) << "\n";
      return std::nullopt;
    }
    const auto *Obj = dyn_cast<ELFObjectFileBase>(BinOrErr->getBinary());
    if (!Obj)
      continue;
    for (ELFSymbolRef Sym : Obj->getDynamicSymbolIterators()) {
      Expected<uint32_t> FlagsOrErr = Sym.getFlags();
      Expected<StringRef> NameOrErr = Sym.getName();
      if (!FlagsOrErr || !NameOrErr) {
        consumeError(FlagsOrErr.takeError());
        consumeError(NameOrErr.takeError());
        return std::nullopt;
      }
      if (Sym.getELFType() == ELF::STT_OBJECT &&
          !(*FlagsOrErr & SymbolRef::SF_Undefined))
        Names.insert(InstrProfSymtab::getCanonicalName(*NameOrErr));
    }
  }
  return Names;
}

Error llvm::generateDataAccessProf(sampleprof::ProfiledBinary &Binary,
                                   StringRef PerfDumpPath, raw_fd_ostream &OS,
                                   std::optional<int32_t> PIDFilter) {
  const auto *ELFObj = dyn_cast<ELFObjectFileBase>(&Binary.getBinary());
  if (!ELFObj)
    return createStringError(inconvertibleErrorCode(),
                             "--memprof-dap requires an ELF binary: " +
                                 Binary.getPath());
  // Data objects are found through .symtab, which stripping removes. Without
  // it every sample would be dropped and the profile would be empty.
  if (ELFObj->symbol_begin() == ELFObj->symbol_end())
    return createStringError(inconvertibleErrorCode(),
                             "--memprof-dap requires a symbol table, but " +
                                 Binary.getPath() + " has none (stripped?)");

  // The preferred ELF VA of every sample in this binary's data. Heap, stack
  // and other files' samples have no preferred VA and stay out of the
  // profile. Appending and sorting once is cheaper than a map insert per
  // sample, and the sorted vector answers the per-symbol range queries.
  std::vector<uint64_t> Addrs;
  bool HasUnresolvedSample = false;
  // True when a non-zero sample belongs to a process that mapped this binary.
  // Preferred VAs from a process with no mmap do not count: another process's
  // mapping must not turn those misses into an all-cold profile.
  bool SawMappedPIDSample = false;
  StringSet<> OtherLoadedFiles;
  if (Error E = forEachCanonicalDataAccessSample(
          &Binary, PerfDumpPath, PIDFilter,
          [&](uint64_t, std::optional<uint64_t> CanonicalDataAddr, uint64_t,
              bool Unresolved, bool MappedPID) {
            if (CanonicalDataAddr)
              Addrs.push_back(*CanonicalDataAddr);
            HasUnresolvedSample |= Unresolved;
            SawMappedPIDSample |= MappedPID;
          },
          &OtherLoadedFiles))
    return E;
  // A mapped image establishes that non-matching samples are genuinely from
  // heap, stack, or another object. Without one, or when every sample belongs
  // to a process that never mapped this binary, no canonicalized sample means
  // the trace may belong to a different binary; do not mark everything cold.
  if (Addrs.empty() &&
      (HasUnresolvedSample || !Binary.hasMappedImage(PIDFilter) ||
       !SawMappedPIDSample))
    return createStringError(
        inconvertibleErrorCode(),
        "No data-access sample maps to " + Binary.getPath() +
            ", and the trace has no usable mapping for that binary");
  llvm::sort(Addrs);

  // Only a shared library's variables can be preempted.
  bool IsSharedLibrary = ELFObj->getEType() == ELF::ET_DYN && !Binary.isPIE();
  std::optional<StringSet<>> DefinedElsewhere;
  if (IsSharedLibrary) {
    // No other ELF was recorded, so an exported symbol may still have been
    // copy-relocated into a file this trace did not map.
    if (!OtherLoadedFiles.empty())
      DefinedElsewhere = getDynamicDataObjects(OtherLoadedFiles);
    if (!DefinedElsewhere)
      WithColor::warning() << "no exported variable of " << Binary.getName()
                           << " is listed as known-cold\n";
  }
  std::vector<DataSymbol> Symbols;
  if (Error E = collectDataSymbols(
          *ELFObj,
          [&](StringRef Name) {
            // Without the other files' symbols, assume any may define it.
            return IsSharedLibrary &&
                   (!DefinedElsewhere || DefinedElsewhere->contains(Name));
          },
          Symbols))
    return E;

  auto DAP = std::make_unique<memprof::DataAccessProfData>();
  if (Error E = buildDataAccessProf(Symbols, Addrs, !HasUnresolvedSample, *DAP))
    return E;

  // An indexed MemProf v4 profile whose only payload is the data-access
  // profile. It has no allocation contexts.
  InstrProfWriter Writer(/*Sparse=*/false,
                         /*TemporalProfTraceReservoirSize=*/0,
                         /*MaxTemporalProfTraceLength=*/0,
                         /*WritePrevVersion=*/false, memprof::Version4);
  if (Error E = Writer.mergeProfileKind(InstrProfKind::MemProf))
    return E;
  Writer.addDataAccessProfData(std::move(DAP));
  return Writer.write(OS);
}
