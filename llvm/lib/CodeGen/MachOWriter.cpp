//===-- MachOWriter.cpp - Target-independent Mach-O Writer code -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-independent Mach-O writer.  This file writes
// out the Mach-O file in the following order:
//
//  #1 FatHeader (universal-only)
//  #2 FatArch (universal-only, 1 per universal arch)
//  Per arch:
//    #3 Header
//    #4 Load Commands
//    #5 Sections
//    #6 Relocations
//    #7 Symbols
//    #8 Strings
//
//===----------------------------------------------------------------------===//

#include "MachOWriter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetJITInfo.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/OutputBuffer.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
using namespace llvm;

/// AddMachOWriter - Concrete function to add the Mach-O writer to the function
/// pass manager.
MachineCodeEmitter *llvm::AddMachOWriter(PassManagerBase &PM,
                                         raw_ostream &O,
                                         TargetMachine &TM) {
  MachOWriter *MOW = new MachOWriter(O, TM);
  PM.add(MOW);
  return &MOW->getMachineCodeEmitter();
}

//===----------------------------------------------------------------------===//
//                       MachOCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {
  /// MachOCodeEmitter - This class is used by the MachOWriter to emit the code 
  /// for functions to the Mach-O file.
  class MachOCodeEmitter : public MachineCodeEmitter {
    MachOWriter &MOW;

    /// Target machine description.
    TargetMachine &TM;

    /// is64Bit/isLittleEndian - This information is inferred from the target
    /// machine directly, indicating what header values and flags to set.
    bool is64Bit, isLittleEndian;

    /// Relocations - These are the relocations that the function needs, as
    /// emitted.
    std::vector<MachineRelocation> Relocations;
    
    /// CPLocations - This is a map of constant pool indices to offsets from the
    /// start of the section for that constant pool index.
    std::vector<uintptr_t> CPLocations;

    /// CPSections - This is a map of constant pool indices to the MachOSection
    /// containing the constant pool entry for that index.
    std::vector<unsigned> CPSections;

    /// JTLocations - This is a map of jump table indices to offsets from the
    /// start of the section for that jump table index.
    std::vector<uintptr_t> JTLocations;

    /// MBBLocations - This vector is a mapping from MBB ID's to their address.
    /// It is filled in by the StartMachineBasicBlock callback and queried by
    /// the getMachineBasicBlockAddress callback.
    std::vector<uintptr_t> MBBLocations;
    
  public:
    MachOCodeEmitter(MachOWriter &mow) : MOW(mow), TM(MOW.TM) {
      is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
      isLittleEndian = TM.getTargetData()->isLittleEndian();
    }

    virtual void startFunction(MachineFunction &MF);
    virtual bool finishFunction(MachineFunction &MF);

    virtual void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }
    
    void emitConstantPool(MachineConstantPool *MCP);
    void emitJumpTables(MachineJumpTableInfo *MJTI);
    
    virtual uintptr_t getConstantPoolEntryAddress(unsigned Index) const {
      assert(CPLocations.size() > Index && "CP not emitted!");
      return CPLocations[Index];
    }
    virtual uintptr_t getJumpTableEntryAddress(unsigned Index) const {
      assert(JTLocations.size() > Index && "JT not emitted!");
      return JTLocations[Index];
    }

    virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
      if (MBBLocations.size() <= (unsigned)MBB->getNumber())
        MBBLocations.resize((MBB->getNumber()+1)*2);
      MBBLocations[MBB->getNumber()] = getCurrentPCOffset();
    }

    virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
      assert(MBBLocations.size() > (unsigned)MBB->getNumber() && 
             MBBLocations[MBB->getNumber()] && "MBB not emitted!");
      return MBBLocations[MBB->getNumber()];
    }

    virtual uintptr_t getLabelAddress(uint64_t Label) const {
      assert(0 && "get Label not implemented");
      abort();
      return 0;
    }

    virtual void emitLabel(uint64_t LabelID) {
      assert(0 && "emit Label not implemented");
      abort();
    }


    virtual void setModuleInfo(llvm::MachineModuleInfo* MMI) { }

    /// JIT SPECIFIC FUNCTIONS - DO NOT IMPLEMENT THESE HERE!
    virtual void startGVStub(const GlobalValue* F, unsigned StubSize,
                             unsigned Alignment = 1) {
      assert(0 && "JIT specific function called!");
      abort();
    }
    virtual void *finishGVStub(const GlobalValue* F) {
      assert(0 && "JIT specific function called!");
      abort();
      return 0;
    }
  };
}

/// startFunction - This callback is invoked when a new machine function is
/// about to be emitted.
void MachOCodeEmitter::startFunction(MachineFunction &MF) {
  const TargetData *TD = TM.getTargetData();
  const Function *F = MF.getFunction();

  // Align the output buffer to the appropriate alignment, power of 2.
  unsigned FnAlign = F->getAlignment();
  unsigned TDAlign = TD->getPrefTypeAlignment(F->getType());
  unsigned Align = Log2_32(std::max(FnAlign, TDAlign));
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");

  // Get the Mach-O Section that this function belongs in.
  MachOWriter::MachOSection *MOS = MOW.getTextSection();
  
  // FIXME: better memory management
  MOS->SectionData.reserve(4096);
  BufferBegin = &MOS->SectionData[0];
  BufferEnd = BufferBegin + MOS->SectionData.capacity();

  // Upgrade the section alignment if required.
  if (MOS->align < Align) MOS->align = Align;

  // Round the size up to the correct alignment for starting the new function.
  if ((MOS->size & ((1 << Align) - 1)) != 0) {
    MOS->size += (1 << Align);
    MOS->size &= ~((1 << Align) - 1);
  }

  // FIXME: Using MOS->size directly here instead of calculating it from the
  // output buffer size (impossible because the code emitter deals only in raw
  // bytes) forces us to manually synchronize size and write padding zero bytes
  // to the output buffer for all non-text sections.  For text sections, we do
  // not synchonize the output buffer, and we just blow up if anyone tries to
  // write non-code to it.  An assert should probably be added to
  // AddSymbolToSection to prevent calling it on the text section.
  CurBufferPtr = BufferBegin + MOS->size;

  // Clear per-function data structures.
  CPLocations.clear();
  CPSections.clear();
  JTLocations.clear();
  MBBLocations.clear();
}

/// finishFunction - This callback is invoked after the function is completely
/// finished.
bool MachOCodeEmitter::finishFunction(MachineFunction &MF) {
  // Get the Mach-O Section that this function belongs in.
  MachOWriter::MachOSection *MOS = MOW.getTextSection();

  // Get a symbol for the function to add to the symbol table
  // FIXME: it seems like we should call something like AddSymbolToSection
  // in startFunction rather than changing the section size and symbol n_value
  // here.
  const GlobalValue *FuncV = MF.getFunction();
  MachOSym FnSym(FuncV, MOW.Mang->getValueName(FuncV), MOS->Index, TM);
  FnSym.n_value = MOS->size;
  MOS->size = CurBufferPtr - BufferBegin;
  
  // Emit constant pool to appropriate section(s)
  emitConstantPool(MF.getConstantPool());

  // Emit jump tables to appropriate section
  emitJumpTables(MF.getJumpTableInfo());
  
  // If we have emitted any relocations to function-specific objects such as 
  // basic blocks, constant pools entries, or jump tables, record their
  // addresses now so that we can rewrite them with the correct addresses
  // later.
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = Relocations[i];
    intptr_t Addr;

    if (MR.isBasicBlock()) {
      Addr = getMachineBasicBlockAddress(MR.getBasicBlock());
      MR.setConstantVal(MOS->Index);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isJumpTableIndex()) {
      Addr = getJumpTableEntryAddress(MR.getJumpTableIndex());
      MR.setConstantVal(MOW.getJumpTableSection()->Index);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isConstantPoolIndex()) {
      Addr = getConstantPoolEntryAddress(MR.getConstantPoolIndex());
      MR.setConstantVal(CPSections[MR.getConstantPoolIndex()]);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isGlobalValue()) {
      // FIXME: This should be a set or something that uniques
      MOW.PendingGlobals.push_back(MR.getGlobalValue());
    } else {
      assert(0 && "Unhandled relocation type");
    }
    MOS->Relocations.push_back(MR);
  }
  Relocations.clear();
  
  // Finally, add it to the symtab.
  MOW.SymbolTable.push_back(FnSym);
  return false;
}

/// emitConstantPool - For each constant pool entry, figure out which section
/// the constant should live in, allocate space for it, and emit it to the 
/// Section data buffer.
void MachOCodeEmitter::emitConstantPool(MachineConstantPool *MCP) {
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty()) return;

  // FIXME: handle PIC codegen
  assert(TM.getRelocationModel() != Reloc::PIC_ &&
         "PIC codegen not yet handled for mach-o jump tables!");

  // Although there is no strict necessity that I am aware of, we will do what
  // gcc for OS X does and put each constant pool entry in a section of constant
  // objects of a certain size.  That means that float constants go in the
  // literal4 section, and double objects go in literal8, etc.
  //
  // FIXME: revisit this decision if we ever do the "stick everything into one
  // "giant object for PIC" optimization.
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    const Type *Ty = CP[i].getType();
    unsigned Size = TM.getTargetData()->getTypePaddedSize(Ty);

    MachOWriter::MachOSection *Sec = MOW.getConstSection(CP[i].Val.ConstVal);
    OutputBuffer SecDataOut(Sec->SectionData, is64Bit, isLittleEndian);

    CPLocations.push_back(Sec->SectionData.size());
    CPSections.push_back(Sec->Index);
    
    // FIXME: remove when we have unified size + output buffer
    Sec->size += Size;

    // Allocate space in the section for the global.
    // FIXME: need alignment?
    // FIXME: share between here and AddSymbolToSection?
    for (unsigned j = 0; j < Size; ++j)
      SecDataOut.outbyte(0);

    MOW.InitMem(CP[i].Val.ConstVal, &Sec->SectionData[0], CPLocations[i],
                TM.getTargetData(), Sec->Relocations);
  }
}

/// emitJumpTables - Emit all the jump tables for a given jump table info
/// record to the appropriate section.
void MachOCodeEmitter::emitJumpTables(MachineJumpTableInfo *MJTI) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  // FIXME: handle PIC codegen
  assert(TM.getRelocationModel() != Reloc::PIC_ &&
         "PIC codegen not yet handled for mach-o jump tables!");

  MachOWriter::MachOSection *Sec = MOW.getJumpTableSection();
  unsigned TextSecIndex = MOW.getTextSection()->Index;
  OutputBuffer SecDataOut(Sec->SectionData, is64Bit, isLittleEndian);

  for (unsigned i = 0, e = JT.size(); i != e; ++i) {
    // For each jump table, record its offset from the start of the section,
    // reserve space for the relocations to the MBBs, and add the relocations.
    const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;
    JTLocations.push_back(Sec->SectionData.size());
    for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi) {
      MachineRelocation MR(MOW.GetJTRelocation(Sec->SectionData.size(),
                                               MBBs[mi]));
      MR.setResultPointer((void *)JTLocations[i]);
      MR.setConstantVal(TextSecIndex);
      Sec->Relocations.push_back(MR);
      SecDataOut.outaddr(0);
    }
  }
  // FIXME: remove when we have unified size + output buffer
  Sec->size = Sec->SectionData.size();
}

//===----------------------------------------------------------------------===//
//                          MachOWriter Implementation
//===----------------------------------------------------------------------===//

char MachOWriter::ID = 0;
MachOWriter::MachOWriter(raw_ostream &o, TargetMachine &tm) 
  : MachineFunctionPass(&ID), O(o), TM(tm) {
  is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
  isLittleEndian = TM.getTargetData()->isLittleEndian();

  // Create the machine code emitter object for this target.
  MCE = new MachOCodeEmitter(*this);
}

MachOWriter::~MachOWriter() {
  delete MCE;
}

void MachOWriter::AddSymbolToSection(MachOSection *Sec, GlobalVariable *GV) {
  const Type *Ty = GV->getType()->getElementType();
  unsigned Size = TM.getTargetData()->getTypePaddedSize(Ty);
  unsigned Align = TM.getTargetData()->getPreferredAlignment(GV);

  // Reserve space in the .bss section for this symbol while maintaining the
  // desired section alignment, which must be at least as much as required by
  // this symbol.
  OutputBuffer SecDataOut(Sec->SectionData, is64Bit, isLittleEndian);

  if (Align) {
    uint64_t OrigSize = Sec->size;
    Align = Log2_32(Align);
    Sec->align = std::max(unsigned(Sec->align), Align);
    Sec->size = (Sec->size + Align - 1) & ~(Align-1);
    
    // Add alignment padding to buffer as well.
    // FIXME: remove when we have unified size + output buffer
    unsigned AlignedSize = Sec->size - OrigSize;
    for (unsigned i = 0; i < AlignedSize; ++i)
      SecDataOut.outbyte(0);
  }
  // Globals without external linkage apparently do not go in the symbol table.
  if (!GV->hasLocalLinkage()) {
    MachOSym Sym(GV, Mang->getValueName(GV), Sec->Index, TM);
    Sym.n_value = Sec->size;
    SymbolTable.push_back(Sym);
  }

  // Record the offset of the symbol, and then allocate space for it.
  // FIXME: remove when we have unified size + output buffer
  Sec->size += Size;
  
  // Now that we know what section the GlovalVariable is going to be emitted 
  // into, update our mappings.
  // FIXME: We may also need to update this when outputting non-GlobalVariable
  // GlobalValues such as functions.
  GVSection[GV] = Sec;
  GVOffset[GV] = Sec->SectionData.size();
  
  // Allocate space in the section for the global.
  for (unsigned i = 0; i < Size; ++i)
    SecDataOut.outbyte(0);
}

void MachOWriter::EmitGlobal(GlobalVariable *GV) {
  const Type *Ty = GV->getType()->getElementType();
  unsigned Size = TM.getTargetData()->getTypePaddedSize(Ty);
  bool NoInit = !GV->hasInitializer();
  
  // If this global has a zero initializer, it is part of the .bss or common
  // section.
  if (NoInit || GV->getInitializer()->isNullValue()) {
    // If this global is part of the common block, add it now.  Variables are
    // part of the common block if they are zero initialized and allowed to be
    // merged with other symbols.
    if (NoInit || GV->hasLinkOnceLinkage() || GV->hasWeakLinkage() ||
        GV->hasCommonLinkage()) {
      MachOSym ExtOrCommonSym(GV, Mang->getValueName(GV), MachOSym::NO_SECT,TM);
      // For undefined (N_UNDF) external (N_EXT) types, n_value is the size in
      // bytes of the symbol.
      ExtOrCommonSym.n_value = Size;
      SymbolTable.push_back(ExtOrCommonSym);
      // Remember that we've seen this symbol
      GVOffset[GV] = Size;
      return;
    }
    // Otherwise, this symbol is part of the .bss section.
    MachOSection *BSS = getBSSSection();
    AddSymbolToSection(BSS, GV);
    return;
  }
  
  // Scalar read-only data goes in a literal section if the scalar is 4, 8, or
  // 16 bytes, or a cstring.  Other read only data goes into a regular const
  // section.  Read-write data goes in the data section.
  MachOSection *Sec = GV->isConstant() ? getConstSection(GV->getInitializer()) : 
                                         getDataSection();
  AddSymbolToSection(Sec, GV);
  InitMem(GV->getInitializer(), &Sec->SectionData[0], GVOffset[GV],
          TM.getTargetData(), Sec->Relocations);
}


bool MachOWriter::runOnMachineFunction(MachineFunction &MF) {
  // Nothing to do here, this is all done through the MCE object.
  return false;
}

bool MachOWriter::doInitialization(Module &M) {
  // Set the magic value, now that we know the pointer size and endianness
  Header.setMagic(isLittleEndian, is64Bit);

  // Set the file type
  // FIXME: this only works for object files, we do not support the creation
  //        of dynamic libraries or executables at this time.
  Header.filetype = MachOHeader::MH_OBJECT;

  Mang = new Mangler(M);
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the Mach-O file to 'O'.
bool MachOWriter::doFinalization(Module &M) {
  // FIXME: we don't handle debug info yet, we should probably do that.

  // Okay, the.text section has been completed, build the .data, .bss, and 
  // "common" sections next.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I);
  
  // Emit the header and load commands.
  EmitHeaderAndLoadCommands();

  // Emit the various sections and their relocation info.
  EmitSections();

  // Write the symbol table and the string table to the end of the file.
  O.write((char*)&SymT[0], SymT.size());
  O.write((char*)&StrT[0], StrT.size());

  // We are done with the abstract symbols.
  SectionList.clear();
  SymbolTable.clear();
  DynamicSymbolTable.clear();

  // Release the name mangler object.
  delete Mang; Mang = 0;
  return false;
}

void MachOWriter::EmitHeaderAndLoadCommands() {
  // Step #0: Fill in the segment load command size, since we need it to figure
  //          out the rest of the header fields
  MachOSegment SEG("", is64Bit);
  SEG.nsects  = SectionList.size();
  SEG.cmdsize = SEG.cmdSize(is64Bit) + 
                SEG.nsects * SectionList[0]->cmdSize(is64Bit);
  
  // Step #1: calculate the number of load commands.  We always have at least
  //          one, for the LC_SEGMENT load command, plus two for the normal
  //          and dynamic symbol tables, if there are any symbols.
  Header.ncmds = SymbolTable.empty() ? 1 : 3;
  
  // Step #2: calculate the size of the load commands
  Header.sizeofcmds = SEG.cmdsize;
  if (!SymbolTable.empty())
    Header.sizeofcmds += SymTab.cmdsize + DySymTab.cmdsize;
    
  // Step #3: write the header to the file
  // Local alias to shortenify coming code.
  DataBuffer &FH = Header.HeaderData;
  OutputBuffer FHOut(FH, is64Bit, isLittleEndian);

  FHOut.outword(Header.magic);
  FHOut.outword(TM.getMachOWriterInfo()->getCPUType());
  FHOut.outword(TM.getMachOWriterInfo()->getCPUSubType());
  FHOut.outword(Header.filetype);
  FHOut.outword(Header.ncmds);
  FHOut.outword(Header.sizeofcmds);
  FHOut.outword(Header.flags);
  if (is64Bit)
    FHOut.outword(Header.reserved);
  
  // Step #4: Finish filling in the segment load command and write it out
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I)
    SEG.filesize += (*I)->size;

  SEG.vmsize = SEG.filesize;
  SEG.fileoff = Header.cmdSize(is64Bit) + Header.sizeofcmds;
  
  FHOut.outword(SEG.cmd);
  FHOut.outword(SEG.cmdsize);
  FHOut.outstring(SEG.segname, 16);
  FHOut.outaddr(SEG.vmaddr);
  FHOut.outaddr(SEG.vmsize);
  FHOut.outaddr(SEG.fileoff);
  FHOut.outaddr(SEG.filesize);
  FHOut.outword(SEG.maxprot);
  FHOut.outword(SEG.initprot);
  FHOut.outword(SEG.nsects);
  FHOut.outword(SEG.flags);
  
  // Step #5: Finish filling in the fields of the MachOSections 
  uint64_t currentAddr = 0;
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    MachOSection *MOS = *I;
    MOS->addr = currentAddr;
    MOS->offset = currentAddr + SEG.fileoff;

    // FIXME: do we need to do something with alignment here?
    currentAddr += MOS->size;
  }
  
  // Step #6: Emit the symbol table to temporary buffers, so that we know the
  // size of the string table when we write the next load command.  This also
  // sorts and assigns indices to each of the symbols, which is necessary for
  // emitting relocations to externally-defined objects.
  BufferSymbolAndStringTable();
  
  // Step #7: Calculate the number of relocations for each section and write out
  // the section commands for each section
  currentAddr += SEG.fileoff;
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    MachOSection *MOS = *I;
    // Convert the relocations to target-specific relocations, and fill in the
    // relocation offset for this section.
    CalculateRelocations(*MOS);
    MOS->reloff = MOS->nreloc ? currentAddr : 0;
    currentAddr += MOS->nreloc * 8;
    
    // write the finalized section command to the output buffer
    FHOut.outstring(MOS->sectname, 16);
    FHOut.outstring(MOS->segname, 16);
    FHOut.outaddr(MOS->addr);
    FHOut.outaddr(MOS->size);
    FHOut.outword(MOS->offset);
    FHOut.outword(MOS->align);
    FHOut.outword(MOS->reloff);
    FHOut.outword(MOS->nreloc);
    FHOut.outword(MOS->flags);
    FHOut.outword(MOS->reserved1);
    FHOut.outword(MOS->reserved2);
    if (is64Bit)
      FHOut.outword(MOS->reserved3);
  }
  
  // Step #8: Emit LC_SYMTAB/LC_DYSYMTAB load commands
  SymTab.symoff  = currentAddr;
  SymTab.nsyms   = SymbolTable.size();
  SymTab.stroff  = SymTab.symoff + SymT.size();
  SymTab.strsize = StrT.size();
  FHOut.outword(SymTab.cmd);
  FHOut.outword(SymTab.cmdsize);
  FHOut.outword(SymTab.symoff);
  FHOut.outword(SymTab.nsyms);
  FHOut.outword(SymTab.stroff);
  FHOut.outword(SymTab.strsize);

  // FIXME: set DySymTab fields appropriately
  // We should probably just update these in BufferSymbolAndStringTable since
  // thats where we're partitioning up the different kinds of symbols.
  FHOut.outword(DySymTab.cmd);
  FHOut.outword(DySymTab.cmdsize);
  FHOut.outword(DySymTab.ilocalsym);
  FHOut.outword(DySymTab.nlocalsym);
  FHOut.outword(DySymTab.iextdefsym);
  FHOut.outword(DySymTab.nextdefsym);
  FHOut.outword(DySymTab.iundefsym);
  FHOut.outword(DySymTab.nundefsym);
  FHOut.outword(DySymTab.tocoff);
  FHOut.outword(DySymTab.ntoc);
  FHOut.outword(DySymTab.modtaboff);
  FHOut.outword(DySymTab.nmodtab);
  FHOut.outword(DySymTab.extrefsymoff);
  FHOut.outword(DySymTab.nextrefsyms);
  FHOut.outword(DySymTab.indirectsymoff);
  FHOut.outword(DySymTab.nindirectsyms);
  FHOut.outword(DySymTab.extreloff);
  FHOut.outword(DySymTab.nextrel);
  FHOut.outword(DySymTab.locreloff);
  FHOut.outword(DySymTab.nlocrel);
  
  O.write((char*)&FH[0], FH.size());
}

/// EmitSections - Now that we have constructed the file header and load
/// commands, emit the data for each section to the file.
void MachOWriter::EmitSections() {
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I)
    // Emit the contents of each section
    O.write((char*)&(*I)->SectionData[0], (*I)->size);
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I)
    // Emit the relocation entry data for each section.
    O.write((char*)&(*I)->RelocBuffer[0], (*I)->RelocBuffer.size());
}

/// PartitionByLocal - Simple boolean predicate that returns true if Sym is
/// a local symbol rather than an external symbol.
bool MachOWriter::PartitionByLocal(const MachOSym &Sym) {
  return (Sym.n_type & (MachOSym::N_EXT | MachOSym::N_PEXT)) == 0;
}

/// PartitionByDefined - Simple boolean predicate that returns true if Sym is
/// defined in this module.
bool MachOWriter::PartitionByDefined(const MachOSym &Sym) {
  // FIXME: Do N_ABS or N_INDR count as defined?
  return (Sym.n_type & MachOSym::N_SECT) == MachOSym::N_SECT;
}

/// BufferSymbolAndStringTable - Sort the symbols we encountered and assign them
/// each a string table index so that they appear in the correct order in the
/// output file.
void MachOWriter::BufferSymbolAndStringTable() {
  // The order of the symbol table is:
  // 1. local symbols
  // 2. defined external symbols (sorted by name)
  // 3. undefined external symbols (sorted by name)
  
  // Before sorting the symbols, check the PendingGlobals for any undefined
  // globals that need to be put in the symbol table.
  for (std::vector<GlobalValue*>::iterator I = PendingGlobals.begin(),
         E = PendingGlobals.end(); I != E; ++I) {
    if (GVOffset[*I] == 0 && GVSection[*I] == 0) {
      MachOSym UndfSym(*I, Mang->getValueName(*I), MachOSym::NO_SECT, TM);
      SymbolTable.push_back(UndfSym);
      GVOffset[*I] = -1;
    }
  }
  
  // Sort the symbols by name, so that when we partition the symbols by scope
  // of definition, we won't have to sort by name within each partition.
  std::sort(SymbolTable.begin(), SymbolTable.end(), MachOSymCmp());

  // Parition the symbol table entries so that all local symbols come before 
  // all symbols with external linkage. { 1 | 2 3 }
  std::partition(SymbolTable.begin(), SymbolTable.end(), PartitionByLocal);
  
  // Advance iterator to beginning of external symbols and partition so that
  // all external symbols defined in this module come before all external
  // symbols defined elsewhere. { 1 | 2 | 3 }
  for (std::vector<MachOSym>::iterator I = SymbolTable.begin(),
         E = SymbolTable.end(); I != E; ++I) {
    if (!PartitionByLocal(*I)) {
      std::partition(I, E, PartitionByDefined);
      break;
    }
  }

  // Calculate the starting index for each of the local, extern defined, and 
  // undefined symbols, as well as the number of each to put in the LC_DYSYMTAB
  // load command.
  for (std::vector<MachOSym>::iterator I = SymbolTable.begin(),
         E = SymbolTable.end(); I != E; ++I) {
    if (PartitionByLocal(*I)) {
      ++DySymTab.nlocalsym;
      ++DySymTab.iextdefsym;
      ++DySymTab.iundefsym;
    } else if (PartitionByDefined(*I)) {
      ++DySymTab.nextdefsym;
      ++DySymTab.iundefsym;
    } else {
      ++DySymTab.nundefsym;
    }
  }
  
  // Write out a leading zero byte when emitting string table, for n_strx == 0
  // which means an empty string.
  OutputBuffer StrTOut(StrT, is64Bit, isLittleEndian);
  StrTOut.outbyte(0);

  // The order of the string table is:
  // 1. strings for external symbols
  // 2. strings for local symbols
  // Since this is the opposite order from the symbol table, which we have just
  // sorted, we can walk the symbol table backwards to output the string table.
  for (std::vector<MachOSym>::reverse_iterator I = SymbolTable.rbegin(),
        E = SymbolTable.rend(); I != E; ++I) {
    if (I->GVName == "") {
      I->n_strx = 0;
    } else {
      I->n_strx = StrT.size();
      StrTOut.outstring(I->GVName, I->GVName.length()+1);
    }
  }

  OutputBuffer SymTOut(SymT, is64Bit, isLittleEndian);

  unsigned index = 0;
  for (std::vector<MachOSym>::iterator I = SymbolTable.begin(),
         E = SymbolTable.end(); I != E; ++I, ++index) {
    // Add the section base address to the section offset in the n_value field
    // to calculate the full address.
    // FIXME: handle symbols where the n_value field is not the address
    GlobalValue *GV = const_cast<GlobalValue*>(I->GV);
    if (GV && GVSection[GV])
      I->n_value += GVSection[GV]->addr;
    if (GV && (GVOffset[GV] == -1))
      GVOffset[GV] = index;
         
    // Emit nlist to buffer
    SymTOut.outword(I->n_strx);
    SymTOut.outbyte(I->n_type);
    SymTOut.outbyte(I->n_sect);
    SymTOut.outhalf(I->n_desc);
    SymTOut.outaddr(I->n_value);
  }
}

/// CalculateRelocations - For each MachineRelocation in the current section,
/// calculate the index of the section containing the object to be relocated,
/// and the offset into that section.  From this information, create the
/// appropriate target-specific MachORelocation type and add buffer it to be
/// written out after we are finished writing out sections.
void MachOWriter::CalculateRelocations(MachOSection &MOS) {
  for (unsigned i = 0, e = MOS.Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = MOS.Relocations[i];
    unsigned TargetSection = MR.getConstantVal();
    unsigned TargetAddr = 0;
    unsigned TargetIndex = 0;

    // This is a scattered relocation entry if it points to a global value with
    // a non-zero offset.
    bool Scattered = false;
    bool Extern = false;

    // Since we may not have seen the GlobalValue we were interested in yet at
    // the time we emitted the relocation for it, fix it up now so that it
    // points to the offset into the correct section.
    if (MR.isGlobalValue()) {
      GlobalValue *GV = MR.getGlobalValue();
      MachOSection *MOSPtr = GVSection[GV];
      intptr_t Offset = GVOffset[GV];
      
      // If we have never seen the global before, it must be to a symbol
      // defined in another module (N_UNDF).
      if (!MOSPtr) {
        // FIXME: need to append stub suffix
        Extern = true;
        TargetAddr = 0;
        TargetIndex = GVOffset[GV];
      } else {
        Scattered = TargetSection != 0;
        TargetSection = MOSPtr->Index;
      }
      MR.setResultPointer((void*)Offset);
    }
    
    // If the symbol is locally defined, pass in the address of the section and
    // the section index to the code which will generate the target relocation.
    if (!Extern) {
        MachOSection &To = *SectionList[TargetSection - 1];
        TargetAddr = To.addr;
        TargetIndex = To.Index;
    }

    OutputBuffer RelocOut(MOS.RelocBuffer, is64Bit, isLittleEndian);
    OutputBuffer SecOut(MOS.SectionData, is64Bit, isLittleEndian);
    
    MOS.nreloc += GetTargetRelocation(MR, MOS.Index, TargetAddr, TargetIndex,
                                      RelocOut, SecOut, Scattered, Extern);
  }
}

// InitMem - Write the value of a Constant to the specified memory location,
// converting it into bytes and relocations.
void MachOWriter::InitMem(const Constant *C, void *Addr, intptr_t Offset,
                          const TargetData *TD, 
                          std::vector<MachineRelocation> &MRs) {
  typedef std::pair<const Constant*, intptr_t> CPair;
  std::vector<CPair> WorkList;
  
  WorkList.push_back(CPair(C,(intptr_t)Addr + Offset));
  
  intptr_t ScatteredOffset = 0;
  
  while (!WorkList.empty()) {
    const Constant *PC = WorkList.back().first;
    intptr_t PA = WorkList.back().second;
    WorkList.pop_back();
    
    if (isa<UndefValue>(PC)) {
      continue;
    } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(PC)) {
      unsigned ElementSize =
        TD->getTypePaddedSize(CP->getType()->getElementType());
      for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
        WorkList.push_back(CPair(CP->getOperand(i), PA+i*ElementSize));
    } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(PC)) {
      //
      // FIXME: Handle ConstantExpression.  See EE::getConstantValue()
      //
      switch (CE->getOpcode()) {
      case Instruction::GetElementPtr: {
        SmallVector<Value*, 8> Indices(CE->op_begin()+1, CE->op_end());
        ScatteredOffset = TD->getIndexedOffset(CE->getOperand(0)->getType(),
                                               &Indices[0], Indices.size());
        WorkList.push_back(CPair(CE->getOperand(0), PA));
        break;
      }
      case Instruction::Add:
      default:
        cerr << "ConstantExpr not handled as global var init: " << *CE << "\n";
        abort();
        break;
      }
    } else if (PC->getType()->isSingleValueType()) {
      unsigned char *ptr = (unsigned char *)PA;
      switch (PC->getType()->getTypeID()) {
      case Type::IntegerTyID: {
        unsigned NumBits = cast<IntegerType>(PC->getType())->getBitWidth();
        uint64_t val = cast<ConstantInt>(PC)->getZExtValue();
        if (NumBits <= 8)
          ptr[0] = val;
        else if (NumBits <= 16) {
          if (TD->isBigEndian())
            val = ByteSwap_16(val);
          ptr[0] = val;
          ptr[1] = val >> 8;
        } else if (NumBits <= 32) {
          if (TD->isBigEndian())
            val = ByteSwap_32(val);
          ptr[0] = val;
          ptr[1] = val >> 8;
          ptr[2] = val >> 16;
          ptr[3] = val >> 24;
        } else if (NumBits <= 64) {
          if (TD->isBigEndian())
            val = ByteSwap_64(val);
          ptr[0] = val;
          ptr[1] = val >> 8;
          ptr[2] = val >> 16;
          ptr[3] = val >> 24;
          ptr[4] = val >> 32;
          ptr[5] = val >> 40;
          ptr[6] = val >> 48;
          ptr[7] = val >> 56;
        } else {
          assert(0 && "Not implemented: bit widths > 64");
        }
        break;
      }
      case Type::FloatTyID: {
        uint32_t val = cast<ConstantFP>(PC)->getValueAPF().bitcastToAPInt().
                        getZExtValue();
        if (TD->isBigEndian())
          val = ByteSwap_32(val);
        ptr[0] = val;
        ptr[1] = val >> 8;
        ptr[2] = val >> 16;
        ptr[3] = val >> 24;
        break;
      }
      case Type::DoubleTyID: {
        uint64_t val = cast<ConstantFP>(PC)->getValueAPF().bitcastToAPInt().
                         getZExtValue();
        if (TD->isBigEndian())
          val = ByteSwap_64(val);
        ptr[0] = val;
        ptr[1] = val >> 8;
        ptr[2] = val >> 16;
        ptr[3] = val >> 24;
        ptr[4] = val >> 32;
        ptr[5] = val >> 40;
        ptr[6] = val >> 48;
        ptr[7] = val >> 56;
        break;
      }
      case Type::PointerTyID:
        if (isa<ConstantPointerNull>(PC))
          memset(ptr, 0, TD->getPointerSize());
        else if (const GlobalValue* GV = dyn_cast<GlobalValue>(PC)) {
          // FIXME: what about function stubs?
          MRs.push_back(MachineRelocation::getGV(PA-(intptr_t)Addr, 
                                                 MachineRelocation::VANILLA,
                                                 const_cast<GlobalValue*>(GV),
                                                 ScatteredOffset));
          ScatteredOffset = 0;
        } else
          assert(0 && "Unknown constant pointer type!");
        break;
      default:
        cerr << "ERROR: Constant unimp for type: " << *PC->getType() << "\n";
        abort();
      }
    } else if (isa<ConstantAggregateZero>(PC)) {
      memset((void*)PA, 0, (size_t)TD->getTypePaddedSize(PC->getType()));
    } else if (const ConstantArray *CPA = dyn_cast<ConstantArray>(PC)) {
      unsigned ElementSize =
        TD->getTypePaddedSize(CPA->getType()->getElementType());
      for (unsigned i = 0, e = CPA->getNumOperands(); i != e; ++i)
        WorkList.push_back(CPair(CPA->getOperand(i), PA+i*ElementSize));
    } else if (const ConstantStruct *CPS = dyn_cast<ConstantStruct>(PC)) {
      const StructLayout *SL =
        TD->getStructLayout(cast<StructType>(CPS->getType()));
      for (unsigned i = 0, e = CPS->getNumOperands(); i != e; ++i)
        WorkList.push_back(CPair(CPS->getOperand(i),
                                 PA+SL->getElementOffset(i)));
    } else {
      cerr << "Bad Type: " << *PC->getType() << "\n";
      assert(0 && "Unknown constant type to initialize memory with!");
    }
  }
}

MachOSym::MachOSym(const GlobalValue *gv, std::string name, uint8_t sect,
                   TargetMachine &TM) :
  GV(gv), n_strx(0), n_type(sect == NO_SECT ? N_UNDF : N_SECT), n_sect(sect),
  n_desc(0), n_value(0) {

  const TargetAsmInfo *TAI = TM.getTargetAsmInfo();  
  
  switch (GV->getLinkage()) {
  default:
    assert(0 && "Unexpected linkage type!");
    break;
  case GlobalValue::WeakLinkage:
  case GlobalValue::LinkOnceLinkage:
  case GlobalValue::CommonLinkage:
    assert(!isa<Function>(gv) && "Unexpected linkage type for Function!");
  case GlobalValue::ExternalLinkage:
    GVName = TAI->getGlobalPrefix() + name;
    n_type |= GV->hasHiddenVisibility() ? N_PEXT : N_EXT;
    break;
  case GlobalValue::PrivateLinkage:
    GVName = TAI->getPrivateGlobalPrefix() + name;
    break;
  case GlobalValue::InternalLinkage:
    GVName = TAI->getGlobalPrefix() + name;
    break;
  }
}
