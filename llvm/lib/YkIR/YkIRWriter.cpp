//===- YkIR/YkIRWRiter.cpp -- Yk JIT IR Serialiaser---------------------===//
//
// Converts an LLVM module into Yk's on-disk AOT IR.
//
//===-------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;
using namespace std;

namespace {

const char *SectionName = ".yk_ir";
const uint32_t Magic = 0xedd5f00d;
const uint32_t Version = 0;

enum OpCode {
  Nop = 0,
};

class YkIRWriter {
private:
  Module &M;
  MCStreamer &OutStreamer;

public:
  YkIRWriter(Module &M, MCStreamer &OutStreamer)
      : M(M), OutStreamer(OutStreamer) {}

  // Serialises a null-terminated string.
  void serialiseString(StringRef S) {
    OutStreamer.emitBinaryData(S);
    OutStreamer.emitInt8(0); // null terminator.
  }

  void serialiseBlock(BasicBlock &BB) {
    // num_instrs:
    OutStreamer.emitSizeT(BB.size());
    // instrs:
    for (Instruction &I : BB) {
      (void)I;
      OutStreamer.emitInt8(OpCode::Nop);
    }
  }

  void serialiseFunc(Function &F) {
    // name:
    serialiseString(F.getName());
    // num_blocks:
    OutStreamer.emitSizeT(F.size());
    // blocks:
    for (BasicBlock &BB : F) {
      serialiseBlock(BB);
    }
  }

  void serialise() {
    // header:
    OutStreamer.emitInt32(Magic);
    OutStreamer.emitInt32(Version);

    // num_funcs:
    OutStreamer.emitSizeT(M.size());
    // funcs:
    for (Function &F : M) {
      serialiseFunc(F);
    }
  }
};
} // anonymous namespace

// Create an ELF section for storing Yk IR into.
MCSection *createYkIRSection(MCContext &Ctx, const MCSection *TextSec) {
  if (Ctx.getObjectFileType() != MCContext::IsELF)
    return nullptr;

  const MCSectionELF *ElfSec = static_cast<const MCSectionELF *>(TextSec);
  unsigned Flags = ELF::SHF_LINK_ORDER;
  StringRef GroupName;

  // Ensure the loader loads it.
  Flags |= ELF::SHF_ALLOC;

  return Ctx.getELFSection(SectionName, ELF::SHT_LLVM_BB_ADDR_MAP, Flags, 0,
                           GroupName, true, ElfSec->getUniqueID(),
                           cast<MCSymbolELF>(TextSec->getBeginSymbol()));
}

// Emit a start/end IR marker.
//
// The JIT uses a start and end marker to make a Rust slice of the IR.
void emitStartOrEndSymbol(MCContext &MCtxt, MCStreamer &OutStreamer,
                          bool Start) {
  std::string SymName("ykllvm.yk_ir.");
  if (Start)
    SymName.append("start");
  else
    SymName.append("stop");

  MCSymbol *Sym = MCtxt.getOrCreateSymbol(SymName);
  OutStreamer.emitSymbolAttribute(Sym, llvm::MCSA_Global);
  OutStreamer.emitLabel(Sym);
}

namespace llvm {

// Emit Yk IR into the resulting ELF binary.
void embedYkIR(MCContext &Ctx, MCStreamer &OutStreamer, Module &M) {
  MCSection *YkIRSec =
      createYkIRSection(Ctx, std::get<0>(OutStreamer.getCurrentSection()));

  OutStreamer.pushSection();
  OutStreamer.switchSection(YkIRSec);
  emitStartOrEndSymbol(Ctx, OutStreamer, true);
  YkIRWriter(M, OutStreamer).serialise();
  emitStartOrEndSymbol(Ctx, OutStreamer, false);
  OutStreamer.popSection();
}
} // namespace llvm
