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

#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
class AsmPrinterHandler;
class BlockAddress;
class ByteStreamer;
class GCStrategy;
class Constant;
class ConstantArray;
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
class MCAsmInfo;
class MCCFIInstruction;
class MCContext;
class MCInst;
class MCInstrInfo;
class MCSection;
class MCStreamer;
class MCSubtargetInfo;
class MCSymbol;
class MDNode;
class DwarfDebug;
class Mangler;
class TargetLoweringObjectFile;
class DataLayout;
class TargetMachine;

/// This class is intended to be used as a driving class for all asm writers.
class AsmPrinter : public MachineFunctionPass {
public:
  /// Target machine description.
  ///
  TargetMachine &TM;

  /// Target Asm Printer information.
  ///
  const MCAsmInfo *MAI;

  const MCInstrInfo *MII;
  /// This is the context for the output file that we are streaming. This owns
  /// all of the global MC-related objects for the generated translation unit.
  MCContext &OutContext;

  /// This is the MCStreamer object for the file we are generating. This
  /// contains the transient state for the current translation unit that we are
  /// generating (such as the current section etc).
  MCStreamer &OutStreamer;

  /// The current machine function.
  const MachineFunction *MF;

  /// This is a pointer to the current MachineModuleInfo.
  MachineModuleInfo *MMI;

  /// Name-mangler for global names.
  ///
  Mangler *Mang;

  /// The symbol for the current function. This is recalculated at the beginning
  /// of each call to runOnMachineFunction().
  ///
  MCSymbol *CurrentFnSym;

  /// The symbol used to represent the start of the current function for the
  /// purpose of calculating its size (e.g. using the .size directive). By
  /// default, this is equal to CurrentFnSym.
  MCSymbol *CurrentFnSymForSize;

private:
  // The garbage collection metadata printer table.
  void *GCMetadataPrinters; // Really a DenseMap.

  /// Emit comments in assembly output if this is true.
  ///
  bool VerboseAsm;
  static char ID;

  /// If VerboseAsm is set, a pointer to the loop info for this function.
  MachineLoopInfo *LI;

  struct HandlerInfo {
    AsmPrinterHandler *Handler;
    const char *TimerName, *TimerGroupName;
    HandlerInfo(AsmPrinterHandler *Handler, const char *TimerName,
                const char *TimerGroupName)
        : Handler(Handler), TimerName(TimerName),
          TimerGroupName(TimerGroupName) {}
  };
  /// A vector of all debug/EH info emitters we should use. This vector
  /// maintains ownership of the emitters.
  SmallVector<HandlerInfo, 1> Handlers;

  /// If the target supports dwarf debug info, this pointer is non-null.
  DwarfDebug *DD;

protected:
  explicit AsmPrinter(TargetMachine &TM, MCStreamer &Streamer);

public:
  virtual ~AsmPrinter();

  DwarfDebug *getDwarfDebug() { return DD; }

  /// Return true if assembly output should contain comments.
  ///
  bool isVerbose() const { return VerboseAsm; }

  /// Return a unique ID for the current function.
  ///
  unsigned getFunctionNumber() const;

  /// Return information about object file lowering.
  const TargetLoweringObjectFile &getObjFileLowering() const;

  /// Return information about data layout.
  const DataLayout &getDataLayout() const;

  /// Return information about subtarget.
  const MCSubtargetInfo &getSubtargetInfo() const;

  void EmitToStreamer(MCStreamer &S, const MCInst &Inst);

  /// Return the target triple string.
  StringRef getTargetTriple() const;

  /// Return the current section we are emitting to.
  const MCSection *getCurrentSection() const;

  void getNameWithPrefix(SmallVectorImpl<char> &Name,
                         const GlobalValue *GV) const;

  MCSymbol *getSymbol(const GlobalValue *GV) const;

  //===------------------------------------------------------------------===//
  // MachineFunctionPass Implementation.
  //===------------------------------------------------------------------===//

  /// Record analysis usage.
  ///
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Set up the AsmPrinter when we are working on a new module. If your pass
  /// overrides this, it must make sure to explicitly call this implementation.
  bool doInitialization(Module &M) override;

  /// Shut down the asmprinter. If you override this in your pass, you must make
  /// sure to call it explicitly.
  bool doFinalization(Module &M) override;

  /// Emit the specified function out to the OutStreamer.
  bool runOnMachineFunction(MachineFunction &MF) override {
    SetupMachineFunction(MF);
    EmitFunctionHeader();
    EmitFunctionBody();
    return false;
  }

  //===------------------------------------------------------------------===//
  // Coarse grained IR lowering routines.
  //===------------------------------------------------------------------===//

  /// This should be called when a new MachineFunction is being processed from
  /// runOnMachineFunction.
  void SetupMachineFunction(MachineFunction &MF);

  /// This method emits the header for the current function.
  void EmitFunctionHeader();

  /// This method emits the body and trailer for a function.
  void EmitFunctionBody();

  void emitCFIInstruction(const MachineInstr &MI);

  enum CFIMoveType { CFI_M_None, CFI_M_EH, CFI_M_Debug };
  CFIMoveType needsCFIMoves();

  bool needsSEHMoves();

  /// Print to the current output stream assembly representations of the
  /// constants in the constant pool MCP. This is used to print out constants
  /// which have been "spilled to memory" by the code generator.
  ///
  virtual void EmitConstantPool();

  /// Print assembly representations of the jump tables used by the current
  /// function to the current output stream.
  ///
  void EmitJumpTableInfo();

  /// Emit the specified global variable to the .s file.
  virtual void EmitGlobalVariable(const GlobalVariable *GV);

  /// Check to see if the specified global is a special global used by LLVM. If
  /// so, emit it and return true, otherwise do nothing and return false.
  bool EmitSpecialLLVMGlobal(const GlobalVariable *GV);

  /// Emit an alignment directive to the specified power of two boundary. For
  /// example, if you pass in 3 here, you will get an 8 byte alignment. If a
  /// global value is specified, and if that global has an explicit alignment
  /// requested, it will override the alignment request if required for
  /// correctness.
  ///
  void EmitAlignment(unsigned NumBits, const GlobalObject *GO = nullptr) const;

  /// This method prints the label for the specified MachineBasicBlock, an
  /// alignment (if present) and a comment describing it if appropriate.
  void EmitBasicBlockStart(const MachineBasicBlock &MBB) const;

  /// \brief Print a general LLVM constant to the .s file.
  void EmitGlobalConstant(const Constant *CV);

  //===------------------------------------------------------------------===//
  // Overridable Hooks
  //===------------------------------------------------------------------===//

  // Targets can, or in the case of EmitInstruction, must implement these to
  // customize output.

  /// This virtual method can be overridden by targets that want to emit
  /// something at the start of their file.
  virtual void EmitStartOfAsmFile(Module &) {}

  /// This virtual method can be overridden by targets that want to emit
  /// something at the end of their file.
  virtual void EmitEndOfAsmFile(Module &) {}

  /// Targets can override this to emit stuff before the first basic block in
  /// the function.
  virtual void EmitFunctionBodyStart() {}

  /// Targets can override this to emit stuff after the last basic block in the
  /// function.
  virtual void EmitFunctionBodyEnd() {}

  /// Targets should implement this to emit instructions.
  virtual void EmitInstruction(const MachineInstr *) {
    llvm_unreachable("EmitInstruction not implemented");
  }

  /// Return the symbol for the specified constant pool entry.
  virtual MCSymbol *GetCPISymbol(unsigned CPID) const;

  virtual void EmitFunctionEntryLabel();

  virtual void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV);

  /// Targets can override this to change how global constants that are part of
  /// a C++ static/global constructor list are emitted.
  virtual void EmitXXStructor(const Constant *CV) { EmitGlobalConstant(CV); }

  /// Return true if the basic block has exactly one predecessor and the control
  /// transfer mechanism between the predecessor and this block is a
  /// fall-through.
  virtual bool
  isBlockOnlyReachableByFallthrough(const MachineBasicBlock *MBB) const;

  /// Targets can override this to customize the output of IMPLICIT_DEF
  /// instructions in verbose mode.
  virtual void emitImplicitDef(const MachineInstr *MI) const;

  //===------------------------------------------------------------------===//
  // Symbol Lowering Routines.
  //===------------------------------------------------------------------===//
public:
  /// Return the MCSymbol corresponding to the assembler temporary label with
  /// the specified stem and unique ID.
  MCSymbol *GetTempSymbol(Twine Name, unsigned ID) const;

  /// Return an assembler temporary label with the specified stem.
  MCSymbol *GetTempSymbol(Twine Name) const;

  /// Return the MCSymbol for a private symbol with global value name as its
  /// base, with the specified suffix.
  MCSymbol *getSymbolWithGlobalValueBase(const GlobalValue *GV,
                                         StringRef Suffix) const;

  /// Return the MCSymbol for the specified ExternalSymbol.
  MCSymbol *GetExternalSymbolSymbol(StringRef Sym) const;

  /// Return the symbol for the specified jump table entry.
  MCSymbol *GetJTISymbol(unsigned JTID, bool isLinkerPrivate = false) const;

  /// Return the symbol for the specified jump table .set
  /// FIXME: privatize to AsmPrinter.
  MCSymbol *GetJTSetSymbol(unsigned UID, unsigned MBBID) const;

  /// Return the MCSymbol used to satisfy BlockAddress uses of the specified
  /// basic block.
  MCSymbol *GetBlockAddressSymbol(const BlockAddress *BA) const;
  MCSymbol *GetBlockAddressSymbol(const BasicBlock *BB) const;

  //===------------------------------------------------------------------===//
  // Emission Helper Routines.
  //===------------------------------------------------------------------===//
public:
  /// This is just convenient handler for printing offsets.
  void printOffset(int64_t Offset, raw_ostream &OS) const;

  /// Emit a byte directive and value.
  ///
  void EmitInt8(int Value) const;

  /// Emit a short directive and value.
  ///
  void EmitInt16(int Value) const;

  /// Emit a long directive and value.
  ///
  void EmitInt32(int Value) const;

  /// Emit something like ".long Hi-Lo" where the size in bytes of the directive
  /// is specified by Size and Hi/Lo specify the labels.  This implicitly uses
  /// .set if it is available.
  void EmitLabelDifference(const MCSymbol *Hi, const MCSymbol *Lo,
                           unsigned Size) const;

  /// Emit something like ".long Hi+Offset-Lo" where the size in bytes of the
  /// directive is specified by Size and Hi/Lo specify the labels.  This
  /// implicitly uses .set if it is available.
  void EmitLabelOffsetDifference(const MCSymbol *Hi, uint64_t Offset,
                                 const MCSymbol *Lo, unsigned Size) const;

  /// Emit something like ".long Label+Offset" where the size in bytes of the
  /// directive is specified by Size and Label specifies the label.  This
  /// implicitly uses .set if it is available.
  void EmitLabelPlusOffset(const MCSymbol *Label, uint64_t Offset,
                           unsigned Size, bool IsSectionRelative = false) const;

  /// Emit something like ".long Label" where the size in bytes of the directive
  /// is specified by Size and Label specifies the label.
  void EmitLabelReference(const MCSymbol *Label, unsigned Size,
                          bool IsSectionRelative = false) const {
    EmitLabelPlusOffset(Label, 0, Size, IsSectionRelative);
  }

  //===------------------------------------------------------------------===//
  // Dwarf Emission Helper Routines
  //===------------------------------------------------------------------===//

  /// Emit the specified signed leb128 value.
  void EmitSLEB128(int64_t Value, const char *Desc = nullptr) const;

  /// Emit the specified unsigned leb128 value.
  void EmitULEB128(uint64_t Value, const char *Desc = nullptr,
                   unsigned PadTo = 0) const;

  /// Emit a .byte 42 directive for a DW_CFA_xxx value.
  void EmitCFAByte(unsigned Val) const;

  /// Emit a .byte 42 directive that corresponds to an encoding.  If verbose
  /// assembly output is enabled, we output comments describing the encoding.
  /// Desc is a string saying what the encoding is specifying (e.g. "LSDA").
  void EmitEncodingByte(unsigned Val, const char *Desc = nullptr) const;

  /// Return the size of the encoding in bytes.
  unsigned GetSizeOfEncodedValue(unsigned Encoding) const;

  /// Emit reference to a ttype global with a specified encoding.
  void EmitTTypeReference(const GlobalValue *GV, unsigned Encoding) const;

  /// Emit the 4-byte offset of Label from the start of its section.  This can
  /// be done with a special directive if the target supports it (e.g. cygwin)
  /// or by emitting it as an offset from a label at the start of the section.
  ///
  /// SectionLabel is a temporary label emitted at the start of the section
  /// that Label lives in.
  void EmitSectionOffset(const MCSymbol *Label,
                         const MCSymbol *SectionLabel) const;

  /// Get the value for DW_AT_APPLE_isa. Zero if no isa encoding specified.
  virtual unsigned getISAEncoding() { return 0; }

  /// \brief Emit a partial DWARF register operation.
  /// \param MLoc             the register
  /// \param PieceSize        size and
  /// \param PieceOffset      offset of the piece in bits, if this is one
  ///                         piece of an aggregate value.
  ///
  /// If size and offset is zero an operation for the entire
  /// register is emitted: Some targets do not provide a DWARF
  /// register number for every register.  If this is the case, this
  /// function will attempt to emit a DWARF register by emitting a
  /// piece of a super-register or by piecing together multiple
  /// subregisters that alias the register.
  void EmitDwarfRegOpPiece(ByteStreamer &BS, const MachineLocation &MLoc,
                           unsigned PieceSize = 0,
                           unsigned PieceOffset = 0) const;

  /// Emit dwarf register operation.
  /// \param Indirect   whether this is a register-indirect address
  virtual void EmitDwarfRegOp(ByteStreamer &BS, const MachineLocation &MLoc,
                              bool Indirect) const;

  //===------------------------------------------------------------------===//
  // Dwarf Lowering Routines
  //===------------------------------------------------------------------===//

  /// \brief Emit frame instruction to describe the layout of the frame.
  void emitCFIInstruction(const MCCFIInstruction &Inst) const;

  //===------------------------------------------------------------------===//
  // Inline Asm Support
  //===------------------------------------------------------------------===//
public:
  // These are hooks that targets can override to implement inline asm
  // support.  These should probably be moved out of AsmPrinter someday.

  /// Print information related to the specified machine instr that is
  /// independent of the operand, and may be independent of the instr itself.
  /// This can be useful for portably encoding the comment character or other
  /// bits of target-specific knowledge into the asmstrings.  The syntax used is
  /// ${:comment}.  Targets can override this to add support for their own
  /// strange codes.
  virtual void PrintSpecial(const MachineInstr *MI, raw_ostream &OS,
                            const char *Code) const;

  /// Print the specified operand of MI, an INLINEASM instruction, using the
  /// specified assembler variant.  Targets should override this to format as
  /// appropriate.  This method can return true if the operand is erroneous.
  virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &OS);

  /// Print the specified operand of MI, an INLINEASM instruction, using the
  /// specified assembler variant as an address. Targets should override this to
  /// format as appropriate.  This method can return true if the operand is
  /// erroneous.
  virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                     unsigned AsmVariant, const char *ExtraCode,
                                     raw_ostream &OS);

  /// Let the target do anything it needs to do after emitting inlineasm.
  /// This callback can be used restore the original mode in case the
  /// inlineasm contains directives to switch modes.
  /// \p StartInfo - the original subtarget info before inline asm
  /// \p EndInfo   - the final subtarget info after parsing the inline asm,
  ///                or NULL if the value is unknown.
  virtual void emitInlineAsmEnd(const MCSubtargetInfo &StartInfo,
                                const MCSubtargetInfo *EndInfo) const;

private:
  /// Private state for PrintSpecial()
  // Assign a unique ID to this machine instruction.
  mutable const MachineInstr *LastMI;
  mutable unsigned LastFn;
  mutable unsigned Counter;
  mutable unsigned SetCounter;

  /// Emit a blob of inline asm to the output streamer.
  void
  EmitInlineAsm(StringRef Str, const MDNode *LocMDNode = nullptr,
                InlineAsm::AsmDialect AsmDialect = InlineAsm::AD_ATT) const;

  /// This method formats and emits the specified machine instruction that is an
  /// inline asm.
  void EmitInlineAsm(const MachineInstr *MI) const;

  //===------------------------------------------------------------------===//
  // Internal Implementation Details
  //===------------------------------------------------------------------===//

  /// This emits visibility information about symbol, if this is suported by the
  /// target.
  void EmitVisibility(MCSymbol *Sym, unsigned Visibility,
                      bool IsDefinition = true) const;

  void EmitLinkage(const GlobalValue *GV, MCSymbol *GVSym) const;

  void EmitJumpTableEntry(const MachineJumpTableInfo *MJTI,
                          const MachineBasicBlock *MBB, unsigned uid) const;
  void EmitLLVMUsedList(const ConstantArray *InitList);
  /// Emit llvm.ident metadata in an '.ident' directive.
  void EmitModuleIdents(Module &M);
  void EmitXXStructorList(const Constant *List, bool isCtor);
  GCMetadataPrinter *GetOrCreateGCPrinter(GCStrategy &C);
};
}

#endif
