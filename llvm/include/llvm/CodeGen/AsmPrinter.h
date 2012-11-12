//===-- llvm/CodeGen/AsmPrinter.h - AsmPrinter Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the base class for target specific
// asm writers.  This class primarily handles common functionality used by
// all asm writers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMPRINTER_H
#define LLVM_CODEGEN_ASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InlineAsm.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
  class BlockAddress;
  class GCStrategy;
  class Constant;
  class GCMetadataPrinter;
  class GlobalValue;
  class GlobalVariable;
  class MachineBasicBlock;
  class MachineFunction;
  class MachineInstr;
  class MachineLocation;
  class MachineLoopInfo;
  class MachineLoop;
  class MachineConstantPoolValue;
  class MachineJumpTableInfo;
  class MachineModuleInfo;
  class MachineMove;
  class MCAsmInfo;
  class MCContext;
  class MCSection;
  class MCStreamer;
  class MCSymbol;
  class MDNode;
  class DwarfDebug;
  class DwarfException;
  class Mangler;
  class TargetLoweringObjectFile;
  class DataLayout;
  class TargetMachine;

  /// AsmPrinter - This class is intended to be used as a driving class for all
  /// asm writers.
  class AsmPrinter : public MachineFunctionPass {
  public:
    /// Target machine description.
    ///
    TargetMachine &TM;

    /// Target Asm Printer information.
    ///
    const MCAsmInfo *MAI;

    /// OutContext - This is the context for the output file that we are
    /// streaming.  This owns all of the global MC-related objects for the
    /// generated translation unit.
    MCContext &OutContext;

    /// OutStreamer - This is the MCStreamer object for the file we are
    /// generating.  This contains the transient state for the current
    /// translation unit that we are generating (such as the current section
    /// etc).
    MCStreamer &OutStreamer;

    /// The current machine function.
    const MachineFunction *MF;

    /// MMI - This is a pointer to the current MachineModuleInfo.
    MachineModuleInfo *MMI;

    /// Name-mangler for global names.
    ///
    Mangler *Mang;

    /// The symbol for the current function. This is recalculated at the
    /// beginning of each call to runOnMachineFunction().
    ///
    MCSymbol *CurrentFnSym;

    /// The symbol used to represent the start of the current function for the
    /// purpose of calculating its size (e.g. using the .size directive). By
    /// default, this is equal to CurrentFnSym.
    MCSymbol *CurrentFnSymForSize;

  private:
    // GCMetadataPrinters - The garbage collection metadata printer table.
    void *GCMetadataPrinters;  // Really a DenseMap.

    /// VerboseAsm - Emit comments in assembly output if this is true.
    ///
    bool VerboseAsm;
    static char ID;

    /// If VerboseAsm is set, a pointer to the loop info for this
    /// function.
    MachineLoopInfo *LI;

    /// DD - If the target supports dwarf debug info, this pointer is non-null.
    DwarfDebug *DD;

    /// DE - If the target supports dwarf exception info, this pointer is
    /// non-null.
    DwarfException *DE;

  protected:
    explicit AsmPrinter(TargetMachine &TM, MCStreamer &Streamer);

  public:
    virtual ~AsmPrinter();

    /// isVerbose - Return true if assembly output should contain comments.
    ///
    bool isVerbose() const { return VerboseAsm; }

    /// getFunctionNumber - Return a unique ID for the current function.
    ///
    unsigned getFunctionNumber() const;

    /// getObjFileLowering - Return information about object file lowering.
    const TargetLoweringObjectFile &getObjFileLowering() const;

    /// getDataLayout - Return information about data layout.
    const DataLayout &getDataLayout() const;

    /// getCurrentSection() - Return the current section we are emitting to.
    const MCSection *getCurrentSection() const;


    //===------------------------------------------------------------------===//
    // MachineFunctionPass Implementation.
    //===------------------------------------------------------------------===//

    /// getAnalysisUsage - Record analysis usage.
    ///
    void getAnalysisUsage(AnalysisUsage &AU) const;

    /// doInitialization - Set up the AsmPrinter when we are working on a new
    /// module.  If your pass overrides this, it must make sure to explicitly
    /// call this implementation.
    bool doInitialization(Module &M);

    /// doFinalization - Shut down the asmprinter.  If you override this in your
    /// pass, you must make sure to call it explicitly.
    bool doFinalization(Module &M);

    /// runOnMachineFunction - Emit the specified function out to the
    /// OutStreamer.
    virtual bool runOnMachineFunction(MachineFunction &MF) {
      SetupMachineFunction(MF);
      EmitFunctionHeader();
      EmitFunctionBody();
      return false;
    }

    //===------------------------------------------------------------------===//
    // Coarse grained IR lowering routines.
    //===------------------------------------------------------------------===//

    /// SetupMachineFunction - This should be called when a new MachineFunction
    /// is being processed from runOnMachineFunction.
    void SetupMachineFunction(MachineFunction &MF);

    /// EmitFunctionHeader - This method emits the header for the current
    /// function.
    void EmitFunctionHeader();

    /// EmitFunctionBody - This method emits the body and trailer for a
    /// function.
    void EmitFunctionBody();

    void emitPrologLabel(const MachineInstr &MI);

    enum CFIMoveType {
      CFI_M_None,
      CFI_M_EH,
      CFI_M_Debug
    };
    CFIMoveType needsCFIMoves();

    bool needsSEHMoves();

    /// needsRelocationsForDwarfStringPool - Specifies whether the object format
    /// expects to use relocations to refer to debug entries. Alternatively we
    /// emit section offsets in bytes from the start of the string pool.
    bool needsRelocationsForDwarfStringPool() const;

    /// EmitConstantPool - Print to the current output stream assembly
    /// representations of the constants in the constant pool MCP. This is
    /// used to print out constants which have been "spilled to memory" by
    /// the code generator.
    ///
    virtual void EmitConstantPool();

    /// EmitJumpTableInfo - Print assembly representations of the jump tables
    /// used by the current function to the current output stream.
    ///
    void EmitJumpTableInfo();

    /// EmitGlobalVariable - Emit the specified global variable to the .s file.
    virtual void EmitGlobalVariable(const GlobalVariable *GV);

    /// EmitSpecialLLVMGlobal - Check to see if the specified global is a
    /// special global used by LLVM.  If so, emit it and return true, otherwise
    /// do nothing and return false.
    bool EmitSpecialLLVMGlobal(const GlobalVariable *GV);

    /// EmitAlignment - Emit an alignment directive to the specified power of
    /// two boundary.  For example, if you pass in 3 here, you will get an 8
    /// byte alignment.  If a global value is specified, and if that global has
    /// an explicit alignment requested, it will override the alignment request
    /// if required for correctness.
    ///
    void EmitAlignment(unsigned NumBits, const GlobalValue *GV = 0) const;

    /// EmitBasicBlockStart - This method prints the label for the specified
    /// MachineBasicBlock, an alignment (if present) and a comment describing
    /// it if appropriate.
    void EmitBasicBlockStart(const MachineBasicBlock *MBB) const;

    /// EmitGlobalConstant - Print a general LLVM constant to the .s file.
    void EmitGlobalConstant(const Constant *CV, unsigned AddrSpace = 0);


    //===------------------------------------------------------------------===//
    // Overridable Hooks
    //===------------------------------------------------------------------===//

    // Targets can, or in the case of EmitInstruction, must implement these to
    // customize output.

    /// EmitStartOfAsmFile - This virtual method can be overridden by targets
    /// that want to emit something at the start of their file.
    virtual void EmitStartOfAsmFile(Module &) {}

    /// EmitEndOfAsmFile - This virtual method can be overridden by targets that
    /// want to emit something at the end of their file.
    virtual void EmitEndOfAsmFile(Module &) {}

    /// EmitFunctionBodyStart - Targets can override this to emit stuff before
    /// the first basic block in the function.
    virtual void EmitFunctionBodyStart() {}

    /// EmitFunctionBodyEnd - Targets can override this to emit stuff after
    /// the last basic block in the function.
    virtual void EmitFunctionBodyEnd() {}

    /// EmitInstruction - Targets should implement this to emit instructions.
    virtual void EmitInstruction(const MachineInstr *) {
      llvm_unreachable("EmitInstruction not implemented");
    }

    virtual void EmitFunctionEntryLabel();

    virtual void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV);

    /// EmitXXStructor - Targets can override this to change how global
    /// constants that are part of a C++ static/global constructor list are
    /// emitted.
    virtual void EmitXXStructor(const Constant *CV) {
      EmitGlobalConstant(CV);
    }

    /// isBlockOnlyReachableByFallthough - Return true if the basic block has
    /// exactly one predecessor and the control transfer mechanism between
    /// the predecessor and this block is a fall-through.
    virtual bool
    isBlockOnlyReachableByFallthrough(const MachineBasicBlock *MBB) const;

    //===------------------------------------------------------------------===//
    // Symbol Lowering Routines.
    //===------------------------------------------------------------------===//
  public:

    /// GetTempSymbol - Return the MCSymbol corresponding to the assembler
    /// temporary label with the specified stem and unique ID.
    MCSymbol *GetTempSymbol(StringRef Name, unsigned ID) const;

    /// GetTempSymbol - Return an assembler temporary label with the specified
    /// stem.
    MCSymbol *GetTempSymbol(StringRef Name) const;


    /// GetSymbolWithGlobalValueBase - Return the MCSymbol for a symbol with
    /// global value name as its base, with the specified suffix, and where the
    /// symbol is forced to have private linkage if ForcePrivate is true.
    MCSymbol *GetSymbolWithGlobalValueBase(const GlobalValue *GV,
                                           StringRef Suffix,
                                           bool ForcePrivate = true) const;

    /// GetExternalSymbolSymbol - Return the MCSymbol for the specified
    /// ExternalSymbol.
    MCSymbol *GetExternalSymbolSymbol(StringRef Sym) const;

    /// GetCPISymbol - Return the symbol for the specified constant pool entry.
    MCSymbol *GetCPISymbol(unsigned CPID) const;

    /// GetJTISymbol - Return the symbol for the specified jump table entry.
    MCSymbol *GetJTISymbol(unsigned JTID, bool isLinkerPrivate = false) const;

    /// GetJTSetSymbol - Return the symbol for the specified jump table .set
    /// FIXME: privatize to AsmPrinter.
    MCSymbol *GetJTSetSymbol(unsigned UID, unsigned MBBID) const;

    /// GetBlockAddressSymbol - Return the MCSymbol used to satisfy BlockAddress
    /// uses of the specified basic block.
    MCSymbol *GetBlockAddressSymbol(const BlockAddress *BA) const;
    MCSymbol *GetBlockAddressSymbol(const BasicBlock *BB) const;

    //===------------------------------------------------------------------===//
    // Emission Helper Routines.
    //===------------------------------------------------------------------===//
  public:
    /// printOffset - This is just convenient handler for printing offsets.
    void printOffset(int64_t Offset, raw_ostream &OS) const;

    /// EmitInt8 - Emit a byte directive and value.
    ///
    void EmitInt8(int Value) const;

    /// EmitInt16 - Emit a short directive and value.
    ///
    void EmitInt16(int Value) const;

    /// EmitInt32 - Emit a long directive and value.
    ///
    void EmitInt32(int Value) const;

    /// EmitLabelDifference - Emit something like ".long Hi-Lo" where the size
    /// in bytes of the directive is specified by Size and Hi/Lo specify the
    /// labels.  This implicitly uses .set if it is available.
    void EmitLabelDifference(const MCSymbol *Hi, const MCSymbol *Lo,
                             unsigned Size) const;

    /// EmitLabelOffsetDifference - Emit something like ".long Hi+Offset-Lo"
    /// where the size in bytes of the directive is specified by Size and Hi/Lo
    /// specify the labels.  This implicitly uses .set if it is available.
    void EmitLabelOffsetDifference(const MCSymbol *Hi, uint64_t Offset,
                                   const MCSymbol *Lo, unsigned Size) const;

    /// EmitLabelPlusOffset - Emit something like ".long Label+Offset"
    /// where the size in bytes of the directive is specified by Size and Label
    /// specifies the label.  This implicitly uses .set if it is available.
    void EmitLabelPlusOffset(const MCSymbol *Label, uint64_t Offset,
                                   unsigned Size) const;

    /// EmitLabelReference - Emit something like ".long Label"
    /// where the size in bytes of the directive is specified by Size and Label
    /// specifies the label.
    void EmitLabelReference(const MCSymbol *Label, unsigned Size) const {
      EmitLabelPlusOffset(Label, 0, Size);
    }

    //===------------------------------------------------------------------===//
    // Dwarf Emission Helper Routines
    //===------------------------------------------------------------------===//

    /// EmitSLEB128 - emit the specified signed leb128 value.
    void EmitSLEB128(int Value, const char *Desc = 0) const;

    /// EmitULEB128 - emit the specified unsigned leb128 value.
    void EmitULEB128(unsigned Value, const char *Desc = 0,
                     unsigned PadTo = 0) const;

    /// EmitCFAByte - Emit a .byte 42 directive for a DW_CFA_xxx value.
    void EmitCFAByte(unsigned Val) const;

    /// EmitEncodingByte - Emit a .byte 42 directive that corresponds to an
    /// encoding.  If verbose assembly output is enabled, we output comments
    /// describing the encoding.  Desc is a string saying what the encoding is
    /// specifying (e.g. "LSDA").
    void EmitEncodingByte(unsigned Val, const char *Desc = 0) const;

    /// GetSizeOfEncodedValue - Return the size of the encoding in bytes.
    unsigned GetSizeOfEncodedValue(unsigned Encoding) const;

    /// EmitReference - Emit a reference to a label with a specified encoding.
    ///
    void EmitReference(const MCSymbol *Sym, unsigned Encoding) const;
    void EmitReference(const GlobalValue *GV, unsigned Encoding) const;

    /// EmitSectionOffset - Emit the 4-byte offset of Label from the start of
    /// its section.  This can be done with a special directive if the target
    /// supports it (e.g. cygwin) or by emitting it as an offset from a label at
    /// the start of the section.
    ///
    /// SectionLabel is a temporary label emitted at the start of the section
    /// that Label lives in.
    void EmitSectionOffset(const MCSymbol *Label,
                           const MCSymbol *SectionLabel) const;

    /// getDebugValueLocation - Get location information encoded by DBG_VALUE
    /// operands.
    virtual MachineLocation getDebugValueLocation(const MachineInstr *MI) const;

    /// getISAEncoding - Get the value for DW_AT_APPLE_isa. Zero if no isa
    /// encoding specified.
    virtual unsigned getISAEncoding() { return 0; }

    /// EmitDwarfRegOp - Emit dwarf register operation.
    virtual void EmitDwarfRegOp(const MachineLocation &MLoc) const;

    //===------------------------------------------------------------------===//
    // Dwarf Lowering Routines
    //===------------------------------------------------------------------===//

    /// EmitCFIFrameMove - Emit frame instruction to describe the layout of the
    /// frame.
    void EmitCFIFrameMove(const MachineMove &Move) const;

    //===------------------------------------------------------------------===//
    // Inline Asm Support
    //===------------------------------------------------------------------===//
  public:
    // These are hooks that targets can override to implement inline asm
    // support.  These should probably be moved out of AsmPrinter someday.

    /// PrintSpecial - Print information related to the specified machine instr
    /// that is independent of the operand, and may be independent of the instr
    /// itself.  This can be useful for portably encoding the comment character
    /// or other bits of target-specific knowledge into the asmstrings.  The
    /// syntax used is ${:comment}.  Targets can override this to add support
    /// for their own strange codes.
    virtual void PrintSpecial(const MachineInstr *MI, raw_ostream &OS,
                              const char *Code) const;

    /// PrintAsmOperand - Print the specified operand of MI, an INLINEASM
    /// instruction, using the specified assembler variant.  Targets should
    /// override this to format as appropriate.  This method can return true if
    /// the operand is erroneous.
    virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode,
                                 raw_ostream &OS);

    /// PrintAsmMemoryOperand - Print the specified operand of MI, an INLINEASM
    /// instruction, using the specified assembler variant as an address.
    /// Targets should override this to format as appropriate.  This method can
    /// return true if the operand is erroneous.
    virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode,
                                       raw_ostream &OS);

  private:
    /// Private state for PrintSpecial()
    // Assign a unique ID to this machine instruction.
    mutable const MachineInstr *LastMI;
    mutable unsigned LastFn;
    mutable unsigned Counter;
    mutable unsigned SetCounter;

    /// EmitInlineAsm - Emit a blob of inline asm to the output streamer.
    void EmitInlineAsm(StringRef Str, const MDNode *LocMDNode = 0,
                    InlineAsm::AsmDialect AsmDialect = InlineAsm::AD_ATT) const;

    /// EmitInlineAsm - This method formats and emits the specified machine
    /// instruction that is an inline asm.
    void EmitInlineAsm(const MachineInstr *MI) const;

    //===------------------------------------------------------------------===//
    // Internal Implementation Details
    //===------------------------------------------------------------------===//

    /// EmitVisibility - This emits visibility information about symbol, if
    /// this is suported by the target.
    void EmitVisibility(MCSymbol *Sym, unsigned Visibility,
                        bool IsDefinition = true) const;

    void EmitLinkage(unsigned Linkage, MCSymbol *GVSym) const;

    void EmitJumpTableEntry(const MachineJumpTableInfo *MJTI,
                            const MachineBasicBlock *MBB,
                            unsigned uid) const;
    void EmitLLVMUsedList(const Constant *List);
    void EmitXXStructorList(const Constant *List, bool isCtor);
    GCMetadataPrinter *GetOrCreateGCPrinter(GCStrategy *C);
  };
}

#endif
