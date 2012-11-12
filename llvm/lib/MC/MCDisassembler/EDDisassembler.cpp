//===-EDDisassembler.cpp - LLVM Enhanced Disassembler ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Enhanced Disassembly library's  disassembler class.
// The disassembler is responsible for vending individual instructions according
// to a given architecture and disassembly syntax.
//
//===----------------------------------------------------------------------===//

#include "EDDisassembler.h"
#include "EDInst.h"
#include "llvm/MC/EDInstInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCTargetAsmLexer.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

EDDisassembler::DisassemblerMap_t EDDisassembler::sDisassemblers;

struct TripleMap {
  Triple::ArchType Arch;
  const char *String;
};

static const struct TripleMap triplemap[] = {
  { Triple::x86,          "i386-unknown-unknown"    },
  { Triple::x86_64,       "x86_64-unknown-unknown"  },
  { Triple::arm,          "arm-unknown-unknown"     },
  { Triple::thumb,        "thumb-unknown-unknown"   }
};

/// infoFromArch - Returns the TripleMap corresponding to a given architecture,
///   or NULL if there is an error
///
/// @arg arch - The Triple::ArchType for the desired architecture
static const char *tripleFromArch(Triple::ArchType arch) {
  unsigned int infoIndex;
  
  for (infoIndex = 0; triplemap[infoIndex].String != NULL; ++infoIndex) {
    if (arch == triplemap[infoIndex].Arch)
      return triplemap[infoIndex].String;
  }
  
  return NULL;
}

/// getLLVMSyntaxVariant - gets the constant to use to get an assembly printer
///   for the desired assembly syntax, suitable for passing to 
///   Target::createMCInstPrinter()
///
/// @arg arch   - The target architecture
/// @arg syntax - The assembly syntax in sd form
static int getLLVMSyntaxVariant(Triple::ArchType arch,
                                EDDisassembler::AssemblySyntax syntax) {
  switch (syntax) {
  // Mappings below from X86AsmPrinter.cpp
  case EDDisassembler::kEDAssemblySyntaxX86ATT:
    if (arch == Triple::x86 || arch == Triple::x86_64)
      return 0;
    break;
  case EDDisassembler::kEDAssemblySyntaxX86Intel:
    if (arch == Triple::x86 || arch == Triple::x86_64)
      return 1;
    break;
  case EDDisassembler::kEDAssemblySyntaxARMUAL:
    if (arch == Triple::arm || arch == Triple::thumb)
      return 0;
    break;
  }

  return -1;
}

EDDisassembler *EDDisassembler::getDisassembler(Triple::ArchType arch,
                                                AssemblySyntax syntax) {
  const char *triple = tripleFromArch(arch);
  return getDisassembler(StringRef(triple), syntax);
}

EDDisassembler *EDDisassembler::getDisassembler(StringRef str,
                                                AssemblySyntax syntax) {
  CPUKey key;
  key.Triple = str.str();
  key.Syntax = syntax;

  EDDisassembler::DisassemblerMap_t::iterator i = sDisassemblers.find(key);

  if (i != sDisassemblers.end()) {
    return i->second;  
  }

  EDDisassembler *sdd = new EDDisassembler(key);
  if (!sdd->valid()) {
    delete sdd;
    return NULL;
  }

  sDisassemblers[key] = sdd;

  return sdd;
}

EDDisassembler::EDDisassembler(CPUKey &key) : 
  Valid(false), 
  HasSemantics(false), 
  ErrorStream(nulls()), 
  Key(key),
  TgtTriple(key.Triple.c_str()) {        
  
  LLVMSyntaxVariant = getLLVMSyntaxVariant(TgtTriple.getArch(), key.Syntax);
  
  if (LLVMSyntaxVariant < 0)
    return;
  
  std::string tripleString(key.Triple);
  std::string errorString;
  
  Tgt = TargetRegistry::lookupTarget(key.Triple, 
                                     errorString);
  
  if (!Tgt)
    return;
  
  MRI.reset(Tgt->createMCRegInfo(tripleString));

  if (!MRI)
    return;

  initMaps(*MRI);
  
  AsmInfo.reset(Tgt->createMCAsmInfo(tripleString));
  
  if (!AsmInfo)
    return;

  STI.reset(Tgt->createMCSubtargetInfo(tripleString, "", ""));
  
  if (!STI)
    return;

  Disassembler.reset(Tgt->createMCDisassembler(*STI));
  
  if (!Disassembler)
    return;
    
  InstInfos = Disassembler->getEDInfo();

  MII.reset(Tgt->createMCInstrInfo());

  if (!MII)
    return;

  InstString.reset(new std::string);
  InstStream.reset(new raw_string_ostream(*InstString));
  InstPrinter.reset(Tgt->createMCInstPrinter(LLVMSyntaxVariant, *AsmInfo,
                                             *MII, *MRI, *STI));
  
  if (!InstPrinter)
    return;
    
  GenericAsmLexer.reset(new AsmLexer(*AsmInfo));
  SpecificAsmLexer.reset(Tgt->createMCAsmLexer(*MRI, *AsmInfo));
  SpecificAsmLexer->InstallLexer(*GenericAsmLexer);
  
  initMaps(*MRI);
    
  Valid = true;
}

EDDisassembler::~EDDisassembler() {
  if (!valid())
    return;
}

namespace {
  /// EDMemoryObject - a subclass of MemoryObject that allows use of a callback
  ///   as provided by the sd interface.  See MemoryObject.
  class EDMemoryObject : public llvm::MemoryObject {
  private:
    EDByteReaderCallback Callback;
    void *Arg;
  public:
    EDMemoryObject(EDByteReaderCallback callback,
                   void *arg) : Callback(callback), Arg(arg) { }
    ~EDMemoryObject() { }
    uint64_t getBase() const { return 0x0; }
    uint64_t getExtent() const { return (uint64_t)-1; }
    int readByte(uint64_t address, uint8_t *ptr) const {
      if (!Callback)
        return -1;
      
      if (Callback(ptr, address, Arg))
        return -1;
      
      return 0;
    }
  };
}

EDInst *EDDisassembler::createInst(EDByteReaderCallback byteReader, 
                                   uint64_t address, 
                                   void *arg) {
  EDMemoryObject memoryObject(byteReader, arg);
  
  MCInst* inst = new MCInst;
  uint64_t byteSize;
  
  MCDisassembler::DecodeStatus S;
  S = Disassembler->getInstruction(*inst, byteSize, memoryObject, address,
                                   ErrorStream, nulls());
  switch (S) {
  case MCDisassembler::Fail:
  case MCDisassembler::SoftFail:
    // FIXME: Do something different on soft failure mode?
    delete inst;
    return NULL;
    
  case MCDisassembler::Success: {
    const llvm::EDInstInfo *thisInstInfo = NULL;

    if (InstInfos) {
      thisInstInfo = &InstInfos[inst->getOpcode()];
    }
    
    EDInst* sdInst = new EDInst(inst, byteSize, *this, thisInstInfo);
    return sdInst;
  }
  }
  return NULL;
}

void EDDisassembler::initMaps(const MCRegisterInfo &registerInfo) {
  unsigned numRegisters = registerInfo.getNumRegs();
  unsigned registerIndex;
  
  for (registerIndex = 0; registerIndex < numRegisters; ++registerIndex) {
    const char* registerName = registerInfo.getName(registerIndex);
    
    RegVec.push_back(registerName);
    RegRMap[registerName] = registerIndex;
  }
  
  switch (TgtTriple.getArch()) {
  default:
    break;
  case Triple::x86:
  case Triple::x86_64:
    stackPointers.insert(registerIDWithName("SP"));
    stackPointers.insert(registerIDWithName("ESP"));
    stackPointers.insert(registerIDWithName("RSP"));
    
    programCounters.insert(registerIDWithName("IP"));
    programCounters.insert(registerIDWithName("EIP"));
    programCounters.insert(registerIDWithName("RIP"));
    break;
  case Triple::arm:
  case Triple::thumb:
    stackPointers.insert(registerIDWithName("SP"));
    
    programCounters.insert(registerIDWithName("PC"));
    break;  
  }
}

const char *EDDisassembler::nameWithRegisterID(unsigned registerID) const {
  if (registerID >= RegVec.size())
    return NULL;
  else
    return RegVec[registerID].c_str();
}

unsigned EDDisassembler::registerIDWithName(const char *name) const {
  regrmap_t::const_iterator iter = RegRMap.find(std::string(name));
  if (iter == RegRMap.end())
    return 0;
  else
    return (*iter).second;
}

bool EDDisassembler::registerIsStackPointer(unsigned registerID) {
  return (stackPointers.find(registerID) != stackPointers.end());
}

bool EDDisassembler::registerIsProgramCounter(unsigned registerID) {
  return (programCounters.find(registerID) != programCounters.end());
}

int EDDisassembler::printInst(std::string &str, MCInst &inst) {
  PrinterMutex.acquire();
  
  InstPrinter->printInst(&inst, *InstStream, "");
  InstStream->flush();
  str = *InstString;
  InstString->clear();
  
  PrinterMutex.release();
  
  return 0;
}

static void diag_handler(const SMDiagnostic &diag, void *context) {
  if (context)
    diag.print("", static_cast<EDDisassembler*>(context)->ErrorStream);
}

int EDDisassembler::parseInst(SmallVectorImpl<MCParsedAsmOperand*> &operands,
                              SmallVectorImpl<AsmToken> &tokens,
                              const std::string &str) {
  int ret = 0;
  
  switch (TgtTriple.getArch()) {
  default:
    return -1;
  case Triple::x86:
  case Triple::x86_64:
  case Triple::arm:
  case Triple::thumb:
    break;
  }
  
  const char *cStr = str.c_str();
  MemoryBuffer *buf = MemoryBuffer::getMemBuffer(cStr, cStr + strlen(cStr));
  
  StringRef instName;
  SMLoc instLoc;
  
  SourceMgr sourceMgr;
  sourceMgr.setDiagHandler(diag_handler, static_cast<void*>(this));
  sourceMgr.AddNewSourceBuffer(buf, SMLoc()); // ownership of buf handed over
  MCContext context(*AsmInfo, *MRI, NULL);
  OwningPtr<MCStreamer> streamer(createNullStreamer(context));
  OwningPtr<MCAsmParser> genericParser(createMCAsmParser(sourceMgr,
                                                         context, *streamer,
                                                         *AsmInfo));

  OwningPtr<MCSubtargetInfo> STI(Tgt->createMCSubtargetInfo(Key.Triple.c_str(), "", ""));
  OwningPtr<MCTargetAsmParser>
    TargetParser(Tgt->createMCAsmParser(*STI, *genericParser));
  
  AsmToken OpcodeToken = genericParser->Lex();
  AsmToken NextToken = genericParser->Lex();  // consume next token, because specificParser expects us to
    
  if (OpcodeToken.is(AsmToken::Identifier)) {
    instName = OpcodeToken.getString();
    instLoc = OpcodeToken.getLoc();
    
    ParseInstructionInfo Info;
    if (NextToken.isNot(AsmToken::Eof) &&
        TargetParser->ParseInstruction(Info, instName, instLoc, operands))
      ret = -1;
  } else {
    ret = -1;
  }
  
  ParserMutex.acquire();
  
  if (!ret) {
    GenericAsmLexer->setBuffer(buf);
  
    while (SpecificAsmLexer->Lex(),
           SpecificAsmLexer->isNot(AsmToken::Eof) &&
           SpecificAsmLexer->isNot(AsmToken::EndOfStatement)) {
      if (SpecificAsmLexer->is(AsmToken::Error)) {
        ret = -1;
        break;
      }
      tokens.push_back(SpecificAsmLexer->getTok());
    }
  }

  ParserMutex.release();
  
  return ret;
}

int EDDisassembler::llvmSyntaxVariant() const {
  return LLVMSyntaxVariant;
}
