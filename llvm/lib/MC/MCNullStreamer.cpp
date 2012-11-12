//===- lib/MC/MCNullStreamer.cpp - Dummy Streamer Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"

using namespace llvm;

namespace {

  class MCNullStreamer : public MCStreamer {
  public:
    MCNullStreamer(MCContext &Context) : MCStreamer(Context) {}

    /// @name MCStreamer Interface
    /// @{

    virtual void InitSections() {
    }

    virtual void ChangeSection(const MCSection *Section) {
    }

    virtual void EmitLabel(MCSymbol *Symbol) {
      assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
      assert(getCurrentSection() && "Cannot emit before setting section!");
      Symbol->setSection(*getCurrentSection());
    }

    virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) {}
    virtual void EmitThumbFunc(MCSymbol *Func) {}

    virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol){}
    virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                          const MCSymbol *LastLabel,
                                          const MCSymbol *Label,
                                          unsigned PointerSize) {}

    virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute){}

    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {}

    virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {}
    virtual void EmitCOFFSymbolStorageClass(int StorageClass) {}
    virtual void EmitCOFFSymbolType(int Type) {}
    virtual void EndCOFFSymbolDef() {}
    virtual void EmitCOFFSecRel32(MCSymbol const *Symbol) {}

    virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                  unsigned ByteAlignment) {}
    virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {}
    virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                              uint64_t Size = 0, unsigned ByteAlignment = 0) {}
    virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment) {}
    virtual void EmitBytes(StringRef Data, unsigned AddrSpace) {}

    virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                               unsigned AddrSpace) {}
    virtual void EmitULEB128Value(const MCExpr *Value) {}
    virtual void EmitSLEB128Value(const MCExpr *Value) {}
    virtual void EmitGPRel32Value(const MCExpr *Value) {}
    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                      unsigned ValueSize = 1,
                                      unsigned MaxBytesToEmit = 0) {}

    virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit = 0) {}

    virtual bool EmitValueToOffset(const MCExpr *Offset,
                                   unsigned char Value = 0) { return false; }

    virtual void EmitFileDirective(StringRef Filename) {}
    virtual bool EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                        StringRef Filename) {
      return false;
    }
    virtual void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                       unsigned Column, unsigned Flags,
                                       unsigned Isa, unsigned Discriminator,
                                       StringRef FileName) {}
    virtual void EmitInstruction(const MCInst &Inst) {}

    virtual void FinishImpl() {}

    virtual void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) {
      RecordProcEnd(Frame);
    }

    /// @}
  };

}

MCStreamer *llvm::createNullStreamer(MCContext &Context) {
  return new MCNullStreamer(Context);
}
