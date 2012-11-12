//===-- MachODump.cpp - Object file dumping utility for llvm --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachO-specific dumper for llvm-objdump.
//
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "MCFunction.h"
#include "llvm/Support/MachO.h"
#include "llvm/Object/MachO.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstring>
using namespace llvm;
using namespace object;

static cl::opt<bool>
  CFG("cfg", cl::desc("Create a CFG for every symbol in the object file and"
                      " write it to a graphviz file (MachO-only)"));

static cl::opt<bool>
  UseDbg("g", cl::desc("Print line information from debug info if available"));

static cl::opt<std::string>
  DSYMFile("dsym", cl::desc("Use .dSYM file for debug info"));

static const Target *GetTarget(const MachOObject *MachOObj) {
  // Figure out the target triple.
  if (TripleName.empty()) {
    llvm::Triple TT("unknown-unknown-unknown");
    switch (MachOObj->getHeader().CPUType) {
    case llvm::MachO::CPUTypeI386:
      TT.setArch(Triple::ArchType(Triple::x86));
      break;
    case llvm::MachO::CPUTypeX86_64:
      TT.setArch(Triple::ArchType(Triple::x86_64));
      break;
    case llvm::MachO::CPUTypeARM:
      TT.setArch(Triple::ArchType(Triple::arm));
      break;
    case llvm::MachO::CPUTypePowerPC:
      TT.setArch(Triple::ArchType(Triple::ppc));
      break;
    case llvm::MachO::CPUTypePowerPC64:
      TT.setArch(Triple::ArchType(Triple::ppc64));
      break;
    }
    TripleName = TT.str();
  }

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget)
    return TheTarget;

  errs() << "llvm-objdump: error: unable to get target for '" << TripleName
         << "', see --version and --triple.\n";
  return 0;
}

struct SymbolSorter {
  bool operator()(const SymbolRef &A, const SymbolRef &B) {
    SymbolRef::Type AType, BType;
    A.getType(AType);
    B.getType(BType);

    uint64_t AAddr, BAddr;
    if (AType != SymbolRef::ST_Function)
      AAddr = 0;
    else
      A.getAddress(AAddr);
    if (BType != SymbolRef::ST_Function)
      BAddr = 0;
    else
      B.getAddress(BAddr);
    return AAddr < BAddr;
  }
};

// Print additional information about an address, if available.
static void DumpAddress(uint64_t Address, ArrayRef<SectionRef> Sections,
                        MachOObject *MachOObj, raw_ostream &OS) {
  for (unsigned i = 0; i != Sections.size(); ++i) {
    uint64_t SectAddr = 0, SectSize = 0;
    Sections[i].getAddress(SectAddr);
    Sections[i].getSize(SectSize);
    uint64_t addr = SectAddr;
    if (SectAddr <= Address &&
        SectAddr + SectSize > Address) {
      StringRef bytes, name;
      Sections[i].getContents(bytes);
      Sections[i].getName(name);
      // Print constant strings.
      if (!name.compare("__cstring"))
        OS << '"' << bytes.substr(addr, bytes.find('\0', addr)) << '"';
      // Print constant CFStrings.
      if (!name.compare("__cfstring"))
        OS << "@\"" << bytes.substr(addr, bytes.find('\0', addr)) << '"';
    }
  }
}

typedef std::map<uint64_t, MCFunction*> FunctionMapTy;
typedef SmallVector<MCFunction, 16> FunctionListTy;
static void createMCFunctionAndSaveCalls(StringRef Name,
                                         const MCDisassembler *DisAsm,
                                         MemoryObject &Object, uint64_t Start,
                                         uint64_t End,
                                         MCInstrAnalysis *InstrAnalysis,
                                         uint64_t Address,
                                         raw_ostream &DebugOut,
                                         FunctionMapTy &FunctionMap,
                                         FunctionListTy &Functions) {
  SmallVector<uint64_t, 16> Calls;
  MCFunction f =
    MCFunction::createFunctionFromMC(Name, DisAsm, Object, Start, End,
                                     InstrAnalysis, DebugOut, Calls);
  Functions.push_back(f);
  FunctionMap[Address] = &Functions.back();

  // Add the gathered callees to the map.
  for (unsigned i = 0, e = Calls.size(); i != e; ++i)
    FunctionMap.insert(std::make_pair(Calls[i], (MCFunction*)0));
}

// Write a graphviz file for the CFG inside an MCFunction.
static void emitDOTFile(const char *FileName, const MCFunction &f,
                        MCInstPrinter *IP) {
  // Start a new dot file.
  std::string Error;
  raw_fd_ostream Out(FileName, Error);
  if (!Error.empty()) {
    errs() << "llvm-objdump: warning: " << Error << '\n';
    return;
  }

  Out << "digraph " << f.getName() << " {\n";
  Out << "graph [ rankdir = \"LR\" ];\n";
  for (MCFunction::iterator i = f.begin(), e = f.end(); i != e; ++i) {
    bool hasPreds = false;
    // Only print blocks that have predecessors.
    // FIXME: Slow.
    for (MCFunction::iterator pi = f.begin(), pe = f.end(); pi != pe;
        ++pi)
      if (pi->second.contains(i->first)) {
        hasPreds = true;
        break;
      }

    if (!hasPreds && i != f.begin())
      continue;

    Out << '"' << i->first << "\" [ label=\"<a>";
    // Print instructions.
    for (unsigned ii = 0, ie = i->second.getInsts().size(); ii != ie;
        ++ii) {
      // Escape special chars and print the instruction in mnemonic form.
      std::string Str;
      raw_string_ostream OS(Str);
      IP->printInst(&i->second.getInsts()[ii].Inst, OS, "");
      Out << DOT::EscapeString(OS.str()) << '|';
    }
    Out << "<o>\" shape=\"record\" ];\n";

    // Add edges.
    for (MCBasicBlock::succ_iterator si = i->second.succ_begin(),
        se = i->second.succ_end(); si != se; ++si)
      Out << i->first << ":o -> " << *si <<":a\n";
  }
  Out << "}\n";
}

static void getSectionsAndSymbols(const macho::Header &Header,
                                  MachOObjectFile *MachOObj,
                             InMemoryStruct<macho::SymtabLoadCommand> *SymtabLC,
                                  std::vector<SectionRef> &Sections,
                                  std::vector<SymbolRef> &Symbols,
                                  SmallVectorImpl<uint64_t> &FoundFns) {
  error_code ec;
  for (symbol_iterator SI = MachOObj->begin_symbols(),
       SE = MachOObj->end_symbols(); SI != SE; SI.increment(ec))
    Symbols.push_back(*SI);

  for (section_iterator SI = MachOObj->begin_sections(),
       SE = MachOObj->end_sections(); SI != SE; SI.increment(ec)) {
    SectionRef SR = *SI;
    StringRef SectName;
    SR.getName(SectName);
    Sections.push_back(*SI);
  }

  for (unsigned i = 0; i != Header.NumLoadCommands; ++i) {
    const MachOObject::LoadCommandInfo &LCI =
       MachOObj->getObject()->getLoadCommandInfo(i);
    if (LCI.Command.Type == macho::LCT_FunctionStarts) {
      // We found a function starts segment, parse the addresses for later
      // consumption.
      InMemoryStruct<macho::LinkeditDataLoadCommand> LLC;
      MachOObj->getObject()->ReadLinkeditDataLoadCommand(LCI, LLC);

      MachOObj->getObject()->ReadULEB128s(LLC->DataOffset, FoundFns);
    }
  }
}

void llvm::DisassembleInputMachO(StringRef Filename) {
  OwningPtr<MemoryBuffer> Buff;

  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
    errs() << "llvm-objdump: " << Filename << ": " << ec.message() << "\n";
    return;
  }

  OwningPtr<MachOObjectFile> MachOOF(static_cast<MachOObjectFile*>(
        ObjectFile::createMachOObjectFile(Buff.take())));
  MachOObject *MachOObj = MachOOF->getObject();

  const Target *TheTarget = GetTarget(MachOObj);
  if (!TheTarget) {
    // GetTarget prints out stuff.
    return;
  }
  OwningPtr<const MCInstrInfo> InstrInfo(TheTarget->createMCInstrInfo());
  OwningPtr<MCInstrAnalysis>
    InstrAnalysis(TheTarget->createMCInstrAnalysis(InstrInfo.get()));

  // Set up disassembler.
  OwningPtr<const MCAsmInfo> AsmInfo(TheTarget->createMCAsmInfo(TripleName));
  OwningPtr<const MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  OwningPtr<const MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI));
  OwningPtr<const MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  OwningPtr<MCInstPrinter>
    IP(TheTarget->createMCInstPrinter(AsmPrinterVariant, *AsmInfo, *InstrInfo,
                                      *MRI, *STI));

  if (!InstrAnalysis || !AsmInfo || !STI || !DisAsm || !IP) {
    errs() << "error: couldn't initialize disassembler for target "
           << TripleName << '\n';
    return;
  }

  outs() << '\n' << Filename << ":\n\n";

  const macho::Header &Header = MachOObj->getHeader();

  const MachOObject::LoadCommandInfo *SymtabLCI = 0;
  // First, find the symbol table segment.
  for (unsigned i = 0; i != Header.NumLoadCommands; ++i) {
    const MachOObject::LoadCommandInfo &LCI = MachOObj->getLoadCommandInfo(i);
    if (LCI.Command.Type == macho::LCT_Symtab) {
      SymtabLCI = &LCI;
      break;
    }
  }

  // Read and register the symbol table data.
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLC;
  if (SymtabLCI) {
    MachOObj->ReadSymtabLoadCommand(*SymtabLCI, SymtabLC);
    MachOObj->RegisterStringTable(*SymtabLC);
  }

  std::vector<SectionRef> Sections;
  std::vector<SymbolRef> Symbols;
  SmallVector<uint64_t, 8> FoundFns;

  getSectionsAndSymbols(Header, MachOOF.get(), &SymtabLC, Sections, Symbols,
                        FoundFns);

  // Make a copy of the unsorted symbol list. FIXME: duplication
  std::vector<SymbolRef> UnsortedSymbols(Symbols);
  // Sort the symbols by address, just in case they didn't come in that way.
  std::sort(Symbols.begin(), Symbols.end(), SymbolSorter());

#ifndef NDEBUG
  raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#else
  raw_ostream &DebugOut = nulls();
#endif

  StringRef DebugAbbrevSection, DebugInfoSection, DebugArangesSection,
            DebugLineSection, DebugStrSection;
  OwningPtr<DIContext> diContext;
  OwningPtr<MachOObjectFile> DSYMObj;
  MachOObject *DbgInfoObj = MachOObj;
  // Try to find debug info and set up the DIContext for it.
  if (UseDbg) {
    ArrayRef<SectionRef> DebugSections = Sections;
    std::vector<SectionRef> DSYMSections;

    // A separate DSym file path was specified, parse it as a macho file,
    // get the sections and supply it to the section name parsing machinery.
    if (!DSYMFile.empty()) {
      OwningPtr<MemoryBuffer> Buf;
      if (error_code ec = MemoryBuffer::getFileOrSTDIN(DSYMFile.c_str(), Buf)) {
        errs() << "llvm-objdump: " << Filename << ": " << ec.message() << '\n';
        return;
      }
      DSYMObj.reset(static_cast<MachOObjectFile*>(
            ObjectFile::createMachOObjectFile(Buf.take())));
      const macho::Header &Header = DSYMObj->getObject()->getHeader();

      std::vector<SymbolRef> Symbols;
      SmallVector<uint64_t, 8> FoundFns;
      getSectionsAndSymbols(Header, DSYMObj.get(), 0, DSYMSections, Symbols,
                            FoundFns);
      DebugSections = DSYMSections;
      DbgInfoObj = DSYMObj.get()->getObject();
    }

    // Find the named debug info sections.
    for (unsigned SectIdx = 0; SectIdx != DebugSections.size(); SectIdx++) {
      StringRef SectName;
      if (!DebugSections[SectIdx].getName(SectName)) {
        if (SectName.equals("__DWARF,__debug_abbrev"))
          DebugSections[SectIdx].getContents(DebugAbbrevSection);
        else if (SectName.equals("__DWARF,__debug_info"))
          DebugSections[SectIdx].getContents(DebugInfoSection);
        else if (SectName.equals("__DWARF,__debug_aranges"))
          DebugSections[SectIdx].getContents(DebugArangesSection);
        else if (SectName.equals("__DWARF,__debug_line"))
          DebugSections[SectIdx].getContents(DebugLineSection);
        else if (SectName.equals("__DWARF,__debug_str"))
          DebugSections[SectIdx].getContents(DebugStrSection);
      }
    }

    // Setup the DIContext.
    diContext.reset(DIContext::getDWARFContext(DbgInfoObj->isLittleEndian(),
                                               DebugInfoSection,
                                               DebugAbbrevSection,
                                               DebugArangesSection,
                                               DebugLineSection,
                                               DebugStrSection));
  }

  FunctionMapTy FunctionMap;
  FunctionListTy Functions;

  for (unsigned SectIdx = 0; SectIdx != Sections.size(); SectIdx++) {
    StringRef SectName;
    if (Sections[SectIdx].getName(SectName) ||
        SectName.compare("__TEXT,__text"))
      continue; // Skip non-text sections

    // Insert the functions from the function starts segment into our map.
    uint64_t VMAddr;
    Sections[SectIdx].getAddress(VMAddr);
    for (unsigned i = 0, e = FoundFns.size(); i != e; ++i) {
      StringRef SectBegin;
      Sections[SectIdx].getContents(SectBegin);
      uint64_t Offset = (uint64_t)SectBegin.data();
      FunctionMap.insert(std::make_pair(VMAddr + FoundFns[i]-Offset,
                                        (MCFunction*)0));
    }

    StringRef Bytes;
    Sections[SectIdx].getContents(Bytes);
    StringRefMemoryObject memoryObject(Bytes);
    bool symbolTableWorked = false;

    // Parse relocations.
    std::vector<std::pair<uint64_t, SymbolRef> > Relocs;
    error_code ec;
    for (relocation_iterator RI = Sections[SectIdx].begin_relocations(),
         RE = Sections[SectIdx].end_relocations(); RI != RE; RI.increment(ec)) {
      uint64_t RelocOffset, SectionAddress;
      RI->getAddress(RelocOffset);
      Sections[SectIdx].getAddress(SectionAddress);
      RelocOffset -= SectionAddress;

      SymbolRef RelocSym;
      RI->getSymbol(RelocSym);

      Relocs.push_back(std::make_pair(RelocOffset, RelocSym));
    }
    array_pod_sort(Relocs.begin(), Relocs.end());

    // Disassemble symbol by symbol.
    for (unsigned SymIdx = 0; SymIdx != Symbols.size(); SymIdx++) {
      StringRef SymName;
      Symbols[SymIdx].getName(SymName);

      SymbolRef::Type ST;
      Symbols[SymIdx].getType(ST);
      if (ST != SymbolRef::ST_Function)
        continue;

      // Make sure the symbol is defined in this section.
      bool containsSym = false;
      Sections[SectIdx].containsSymbol(Symbols[SymIdx], containsSym);
      if (!containsSym)
        continue;

      // Start at the address of the symbol relative to the section's address.
      uint64_t SectionAddress = 0;
      uint64_t Start = 0;
      Sections[SectIdx].getAddress(SectionAddress);
      Symbols[SymIdx].getAddress(Start);
      Start -= SectionAddress;

      // Stop disassembling either at the beginning of the next symbol or at
      // the end of the section.
      bool containsNextSym = false;
      uint64_t NextSym = 0;
      uint64_t NextSymIdx = SymIdx+1;
      while (Symbols.size() > NextSymIdx) {
        SymbolRef::Type NextSymType;
        Symbols[NextSymIdx].getType(NextSymType);
        if (NextSymType == SymbolRef::ST_Function) {
          Sections[SectIdx].containsSymbol(Symbols[NextSymIdx],
                                           containsNextSym);
          Symbols[NextSymIdx].getAddress(NextSym);
          NextSym -= SectionAddress;
          break;
        }
        ++NextSymIdx;
      }

      uint64_t SectSize;
      Sections[SectIdx].getSize(SectSize);
      uint64_t End = containsNextSym ?  NextSym : SectSize;
      uint64_t Size;

      symbolTableWorked = true;

      if (!CFG) {
        // Normal disassembly, print addresses, bytes and mnemonic form.
        StringRef SymName;
        Symbols[SymIdx].getName(SymName);

        outs() << SymName << ":\n";
        DILineInfo lastLine;
        for (uint64_t Index = Start; Index < End; Index += Size) {
          MCInst Inst;

          if (DisAsm->getInstruction(Inst, Size, memoryObject, Index,
                                     DebugOut, nulls())) {
            uint64_t SectAddress = 0;
            Sections[SectIdx].getAddress(SectAddress);
            outs() << format("%8" PRIx64 ":\t", SectAddress + Index);

            DumpBytes(StringRef(Bytes.data() + Index, Size));
            IP->printInst(&Inst, outs(), "");

            // Print debug info.
            if (diContext) {
              DILineInfo dli =
                diContext->getLineInfoForAddress(SectAddress + Index);
              // Print valid line info if it changed.
              if (dli != lastLine && dli.getLine() != 0)
                outs() << "\t## " << dli.getFileName() << ':'
                       << dli.getLine() << ':' << dli.getColumn();
              lastLine = dli;
            }
            outs() << "\n";
          } else {
            errs() << "llvm-objdump: warning: invalid instruction encoding\n";
            if (Size == 0)
              Size = 1; // skip illegible bytes
          }
        }
      } else {
        // Create CFG and use it for disassembly.
        StringRef SymName;
        Symbols[SymIdx].getName(SymName);
        createMCFunctionAndSaveCalls(
            SymName, DisAsm.get(), memoryObject, Start, End,
            InstrAnalysis.get(), Start, DebugOut, FunctionMap, Functions);
      }
    }
    if (!CFG && !symbolTableWorked) {
      // Reading the symbol table didn't work, disassemble the whole section. 
      uint64_t SectAddress;
      Sections[SectIdx].getAddress(SectAddress);
      uint64_t SectSize;
      Sections[SectIdx].getSize(SectSize);
      uint64_t InstSize;
      for (uint64_t Index = 0; Index < SectSize; Index += InstSize) {
        MCInst Inst;

        if (DisAsm->getInstruction(Inst, InstSize, memoryObject, Index,
                                   DebugOut, nulls())) {
          outs() << format("%8" PRIx64 ":\t", SectAddress + Index);
          DumpBytes(StringRef(Bytes.data() + Index, InstSize));
          IP->printInst(&Inst, outs(), "");
          outs() << "\n";
        } else {
          errs() << "llvm-objdump: warning: invalid instruction encoding\n";
          if (InstSize == 0)
            InstSize = 1; // skip illegible bytes
        }
      }
    }

    if (CFG) {
      if (!symbolTableWorked) {
        // Reading the symbol table didn't work, create a big __TEXT symbol.
        uint64_t SectSize = 0, SectAddress = 0;
        Sections[SectIdx].getSize(SectSize);
        Sections[SectIdx].getAddress(SectAddress);
        createMCFunctionAndSaveCalls("__TEXT", DisAsm.get(), memoryObject,
                                     0, SectSize,
                                     InstrAnalysis.get(),
                                     SectAddress, DebugOut,
                                     FunctionMap, Functions);
      }
      for (std::map<uint64_t, MCFunction*>::iterator mi = FunctionMap.begin(),
           me = FunctionMap.end(); mi != me; ++mi)
        if (mi->second == 0) {
          // Create functions for the remaining callees we have gathered,
          // but we didn't find a name for them.
          uint64_t SectSize = 0;
          Sections[SectIdx].getSize(SectSize);

          SmallVector<uint64_t, 16> Calls;
          MCFunction f =
            MCFunction::createFunctionFromMC("unknown", DisAsm.get(),
                                             memoryObject, mi->first,
                                             SectSize,
                                             InstrAnalysis.get(), DebugOut,
                                             Calls);
          Functions.push_back(f);
          mi->second = &Functions.back();
          for (unsigned i = 0, e = Calls.size(); i != e; ++i) {
            std::pair<uint64_t, MCFunction*> p(Calls[i], (MCFunction*)0);
            if (FunctionMap.insert(p).second)
              mi = FunctionMap.begin();
          }
        }

      DenseSet<uint64_t> PrintedBlocks;
      for (unsigned ffi = 0, ffe = Functions.size(); ffi != ffe; ++ffi) {
        MCFunction &f = Functions[ffi];
        for (MCFunction::iterator fi = f.begin(), fe = f.end(); fi != fe; ++fi){
          if (!PrintedBlocks.insert(fi->first).second)
            continue; // We already printed this block.

          // We assume a block has predecessors when it's the first block after
          // a symbol.
          bool hasPreds = FunctionMap.find(fi->first) != FunctionMap.end();

          // See if this block has predecessors.
          // FIXME: Slow.
          for (MCFunction::iterator pi = f.begin(), pe = f.end(); pi != pe;
              ++pi)
            if (pi->second.contains(fi->first)) {
              hasPreds = true;
              break;
            }

          uint64_t SectSize = 0, SectAddress;
          Sections[SectIdx].getSize(SectSize);
          Sections[SectIdx].getAddress(SectAddress);

          // No predecessors, this is a data block. Print as .byte directives.
          if (!hasPreds) {
            uint64_t End = llvm::next(fi) == fe ? SectSize :
                                                  llvm::next(fi)->first;
            outs() << "# " << End-fi->first << " bytes of data:\n";
            for (unsigned pos = fi->first; pos != End; ++pos) {
              outs() << format("%8x:\t", SectAddress + pos);
              DumpBytes(StringRef(Bytes.data() + pos, 1));
              outs() << format("\t.byte 0x%02x\n", (uint8_t)Bytes[pos]);
            }
            continue;
          }

          if (fi->second.contains(fi->first)) // Print a header for simple loops
            outs() << "# Loop begin:\n";

          DILineInfo lastLine;
          // Walk over the instructions and print them.
          for (unsigned ii = 0, ie = fi->second.getInsts().size(); ii != ie;
               ++ii) {
            const MCDecodedInst &Inst = fi->second.getInsts()[ii];

            // If there's a symbol at this address, print its name.
            if (FunctionMap.find(SectAddress + Inst.Address) !=
                FunctionMap.end())
              outs() << FunctionMap[SectAddress + Inst.Address]-> getName()
                     << ":\n";

            outs() << format("%8" PRIx64 ":\t", SectAddress + Inst.Address);
            DumpBytes(StringRef(Bytes.data() + Inst.Address, Inst.Size));

            if (fi->second.contains(fi->first)) // Indent simple loops.
              outs() << '\t';

            IP->printInst(&Inst.Inst, outs(), "");

            // Look for relocations inside this instructions, if there is one
            // print its target and additional information if available.
            for (unsigned j = 0; j != Relocs.size(); ++j)
              if (Relocs[j].first >= SectAddress + Inst.Address &&
                  Relocs[j].first < SectAddress + Inst.Address + Inst.Size) {
                StringRef SymName;
                uint64_t Addr;
                Relocs[j].second.getAddress(Addr);
                Relocs[j].second.getName(SymName);

                outs() << "\t# " << SymName << ' ';
                DumpAddress(Addr, Sections, MachOObj, outs());
              }

            // If this instructions contains an address, see if we can evaluate
            // it and print additional information.
            uint64_t targ = InstrAnalysis->evaluateBranch(Inst.Inst,
                                                          Inst.Address,
                                                          Inst.Size);
            if (targ != -1ULL)
              DumpAddress(targ, Sections, MachOObj, outs());

            // Print debug info.
            if (diContext) {
              DILineInfo dli =
                diContext->getLineInfoForAddress(SectAddress + Inst.Address);
              // Print valid line info if it changed.
              if (dli != lastLine && dli.getLine() != 0)
                outs() << "\t## " << dli.getFileName() << ':'
                       << dli.getLine() << ':' << dli.getColumn();
              lastLine = dli;
            }

            outs() << '\n';
          }
        }

        emitDOTFile((f.getName().str() + ".dot").c_str(), f, IP.get());
      }
    }
  }
}
