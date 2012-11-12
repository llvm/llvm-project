//===-- NVPTXAsmPrinter.h - NVPTX LLVM assembly writer --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to NVPTX assembly language.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXASMPRINTER_H
#define NVPTXASMPRINTER_H

#include "NVPTX.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXSubtarget.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/Mangler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include <fstream>

// The ptx syntax and format is very different from that usually seem in a .s
// file,
// therefore we are not able to use the MCAsmStreamer interface here.
//
// We are handcrafting the output method here.
//
// A better approach is to clone the MCAsmStreamer to a MCPTXAsmStreamer
// (subclass of MCStreamer).

// This is defined in AsmPrinter.cpp.
// Used to process the constant expressions in initializers.
namespace nvptx {
const llvm::MCExpr *LowerConstant(const llvm::Constant *CV,
                                  llvm::AsmPrinter &AP) ;
}

namespace llvm {

class LineReader {
private:
  unsigned theCurLine ;
  std::ifstream fstr;
  char buff[512];
  std::string theFileName;
  SmallVector<unsigned, 32> lineOffset;
public:
  LineReader(std::string filename) {
    theCurLine = 0;
    fstr.open(filename.c_str());
    theFileName = filename;
  }
  std::string fileName() { return theFileName; }
  ~LineReader() {
    fstr.close();
  }
  std::string readLine(unsigned line);
};



class LLVM_LIBRARY_VISIBILITY NVPTXAsmPrinter : public AsmPrinter {


  class AggBuffer {
    // Used to buffer the emitted string for initializing global
    // aggregates.
    //
    // Normally an aggregate (array, vector or structure) is emitted
    // as a u8[]. However, if one element/field of the aggregate
    // is a non-NULL address, then the aggregate is emitted as u32[]
    // or u64[].
    //
    // We first layout the aggregate in 'buffer' in bytes, except for
    // those symbol addresses. For the i-th symbol address in the
    //aggregate, its corresponding 4-byte or 8-byte elements in 'buffer'
    // are filled with 0s. symbolPosInBuffer[i-1] records its position
    // in 'buffer', and Symbols[i-1] records the Value*.
    //
    // Once we have this AggBuffer setup, we can choose how to print
    // it out.
  public:
    unsigned size;   // size of the buffer in bytes
    unsigned char *buffer; // the buffer
    unsigned numSymbols;   // number of symbol addresses
    SmallVector<unsigned, 4> symbolPosInBuffer;
    SmallVector<Value *, 4> Symbols;

  private:
    unsigned curpos;
    raw_ostream &O;
    NVPTXAsmPrinter &AP;

  public:
    AggBuffer(unsigned _size, raw_ostream &_O, NVPTXAsmPrinter &_AP)
    :O(_O),AP(_AP) {
      buffer = new unsigned char[_size];
      size = _size;
      curpos = 0;
      numSymbols = 0;
    }
    ~AggBuffer() {
      delete [] buffer;
    }
    unsigned addBytes(unsigned char *Ptr, int Num, int Bytes) {
      assert((curpos+Num) <= size);
      assert((curpos+Bytes) <= size);
      for ( int i= 0; i < Num; ++i) {
        buffer[curpos] = Ptr[i];
        curpos ++;
      }
      for ( int i=Num; i < Bytes ; ++i) {
        buffer[curpos] = 0;
        curpos ++;
      }
      return curpos;
    }
    unsigned addZeros(int Num) {
      assert((curpos+Num) <= size);
      for ( int i= 0; i < Num; ++i) {
        buffer[curpos] = 0;
        curpos ++;
      }
      return curpos;
    }
    void addSymbol(Value *GVar) {
      symbolPosInBuffer.push_back(curpos);
      Symbols.push_back(GVar);
      numSymbols++;
    }
    void print() {
      if (numSymbols == 0) {
        // print out in bytes
        for (unsigned i=0; i<size; i++) {
          if (i)
            O << ", ";
          O << (unsigned int)buffer[i];
        }
      } else {
        // print out in 4-bytes or 8-bytes
        unsigned int pos = 0;
        unsigned int nSym = 0;
        unsigned int nextSymbolPos = symbolPosInBuffer[nSym];
        unsigned int nBytes = 4;
        if (AP.nvptxSubtarget.is64Bit())
          nBytes = 8;
        for (pos=0; pos<size; pos+=nBytes) {
          if (pos)
            O << ", ";
          if (pos == nextSymbolPos) {
            Value *v = Symbols[nSym];
            if (GlobalValue *GVar = dyn_cast<GlobalValue>(v)) {
              MCSymbol *Name = AP.Mang->getSymbol(GVar);
              O << *Name;
            }
            else if (ConstantExpr *Cexpr =
                dyn_cast<ConstantExpr>(v)) {
              O << *nvptx::LowerConstant(Cexpr, AP);
            } else
              llvm_unreachable("symbol type unknown");
            nSym++;
            if (nSym >= numSymbols)
              nextSymbolPos = size+1;
            else
              nextSymbolPos = symbolPosInBuffer[nSym];
          } else
            if (nBytes == 4)
              O << *(unsigned int*)(buffer+pos);
            else
              O << *(unsigned long long*)(buffer+pos);
        }
      }
    }
  };

  friend class AggBuffer;

  virtual void emitSrcInText(StringRef filename, unsigned line);

private :
  virtual const char *getPassName() const {
    return "NVPTX Assembly Printer";
  }

  const Function *F;
  std::string CurrentFnName;

  void EmitFunctionEntryLabel();
  void EmitFunctionBodyStart();
  void EmitFunctionBodyEnd();

  void EmitInstruction(const MachineInstr *);

  void EmitAlignment(unsigned NumBits, const GlobalValue *GV = 0) const {}

  void printGlobalVariable(const GlobalVariable *GVar);
  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                    const char *Modifier=0);
  void printLdStCode(const MachineInstr *MI, int opNum, raw_ostream &O,
                     const char *Modifier=0);
  void printVecModifiedImmediate(const MachineOperand &MO,
                                 const char *Modifier, raw_ostream &O);
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                       const char *Modifier=0);
  void printImplicitDef(const MachineInstr *MI, raw_ostream &O) const;
  // definition autogenerated.
  void printInstruction(const MachineInstr *MI, raw_ostream &O);
  void printModuleLevelGV(GlobalVariable* GVar, raw_ostream &O,
                          bool=false);
  void printParamName(int paramIndex, raw_ostream &O);
  void printParamName(Function::const_arg_iterator I, int paramIndex,
                      raw_ostream &O);
  void emitHeader(Module &M, raw_ostream &O);
  void emitKernelFunctionDirectives(const Function& F,
                                    raw_ostream &O) const;
  void emitVirtualRegister(unsigned int vr, bool isVec, raw_ostream &O);
  void emitFunctionExternParamList(const MachineFunction &MF);
  void emitFunctionParamList(const Function *, raw_ostream &O);
  void emitFunctionParamList(const MachineFunction &MF, raw_ostream &O);
  void setAndEmitFunctionVirtualRegisters(const MachineFunction &MF);
  void emitFunctionTempData(const MachineFunction &MF,
                            unsigned &FrameSize);
  bool isImageType(const Type *Ty);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &);
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &);
  void printReturnValStr(const Function *, raw_ostream &O);
  void printReturnValStr(const MachineFunction &MF, raw_ostream &O);

protected:
  bool doInitialization(Module &M);
  bool doFinalization(Module &M);

private:
  std::string CurrentBankselLabelInBasicBlock;

  // This is specific per MachineFunction.
  const MachineRegisterInfo *MRI;
  // The contents are specific for each
  // MachineFunction. But the size of the
  // array is not.
  std::map<unsigned, unsigned> *VRidGlobal2LocalMap;
  // cache the subtarget here.
  const NVPTXSubtarget &nvptxSubtarget;
  // Build the map between type name and ID based on module's type
  // symbol table.
  std::map<const Type *, std::string> TypeNameMap;

  // List of variables demoted to a function scope.
  std::map<const Function *, std::vector<GlobalVariable *> > localDecls;

  // To record filename to ID mapping
  std::map<std::string, unsigned> filenameMap;
  void recordAndEmitFilenames(Module &);

  void emitPTXGlobalVariable(const GlobalVariable *GVar, raw_ostream &O);
  void emitPTXAddressSpace(unsigned int AddressSpace,
                           raw_ostream &O) const;
  std::string getPTXFundamentalTypeStr(const Type *Ty, bool=true) const ;
  void printScalarConstant(Constant *CPV, raw_ostream &O) ;
  void printFPConstant(const ConstantFP *Fp, raw_ostream &O) ;
  void bufferLEByte(Constant *CPV, int Bytes, AggBuffer *aggBuffer) ;
  void bufferAggregateConstant(Constant *CV, AggBuffer *aggBuffer) ;

  void printOperandProper(const MachineOperand &MO);

  void emitLinkageDirective(const GlobalValue* V, raw_ostream &O);
  void emitDeclarations(Module &, raw_ostream &O);
  void emitDeclaration(const Function *, raw_ostream &O);

  static const char *getRegisterName(unsigned RegNo);
  void emitDemotedVars(const Function *, raw_ostream &);

  LineReader *reader;
  LineReader *getReader(std::string);
public:
  NVPTXAsmPrinter(TargetMachine &TM,
                  MCStreamer &Streamer)
  : AsmPrinter(TM, Streamer),
    nvptxSubtarget(TM.getSubtarget<NVPTXSubtarget>()) {
    CurrentBankselLabelInBasicBlock = "";
    VRidGlobal2LocalMap = NULL;
    reader = NULL;
  }

  ~NVPTXAsmPrinter() {
    if (!reader)
      delete reader;
  }

  bool ignoreLoc(const MachineInstr &);

  virtual void getVirtualRegisterName(unsigned, bool, raw_ostream &);

  DebugLoc prevDebugLoc;
  void emitLineNumberAsDotLoc(const MachineInstr &);
};
} // end of namespace

#endif
