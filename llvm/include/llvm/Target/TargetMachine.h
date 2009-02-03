//===-- llvm/Target/TargetMachine.h - Target Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetMachine and LLVMTargetMachine classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINE_H
#define LLVM_TARGET_TARGETMACHINE_H

#include "llvm/Target/TargetInstrItineraries.h"
#include <cassert>

namespace llvm {

class TargetAsmInfo;
class TargetData;
class TargetSubtarget;
class TargetInstrInfo;
class TargetJITInfo;
class TargetLowering;
class TargetFrameInfo;
class MachineCodeEmitter;
class TargetRegisterInfo;
class Module;
class PassManagerBase;
class PassManager;
class Pass;
class TargetMachOWriterInfo;
class TargetELFWriterInfo;
class raw_ostream;

// Relocation model types.
namespace Reloc {
  enum Model {
    Default,
    Static,
    PIC_,         // Cannot be named PIC due to collision with -DPIC
    DynamicNoPIC
  };
}

// Code model types.
namespace CodeModel {
  enum Model {
    Default,
    Small,
    Kernel,
    Medium,
    Large
  };
}

namespace FileModel {
  enum Model {
    Error,
    None,
    AsmFile,
    MachOFile,
    ElfFile
  };
}

//===----------------------------------------------------------------------===//
///
/// TargetMachine - Primary interface to the complete machine description for
/// the target machine.  All target-specific information should be accessible
/// through this interface.
///
class TargetMachine {
  TargetMachine(const TargetMachine &);   // DO NOT IMPLEMENT
  void operator=(const TargetMachine &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetMachine() : AsmInfo(0) { }

  /// getSubtargetImpl - virtual method implemented by subclasses that returns
  /// a reference to that target's TargetSubtarget-derived member variable.
  virtual const TargetSubtarget *getSubtargetImpl() const { return 0; }
  
  /// AsmInfo - Contains target specific asm information.
  ///
  mutable const TargetAsmInfo *AsmInfo;
  
  /// createTargetAsmInfo - Create a new instance of target specific asm
  /// information.
  virtual const TargetAsmInfo *createTargetAsmInfo() const { return 0; }

public:
  virtual ~TargetMachine();

  /// getModuleMatchQuality - This static method should be implemented by
  /// targets to indicate how closely they match the specified module.  This is
  /// used by the LLC tool to determine which target to use when an explicit
  /// -march option is not specified.  If a target returns zero, it will never
  /// be chosen without an explicit -march option.
  static unsigned getModuleMatchQuality(const Module &) { return 0; }

  /// getJITMatchQuality - This static method should be implemented by targets
  /// that provide JIT capabilities to indicate how suitable they are for
  /// execution on the current host.  If a value of 0 is returned, the target
  /// will not be used unless an explicit -march option is used.
  static unsigned getJITMatchQuality() { return 0; }

  // Interfaces to the major aspects of target machine information:
  // -- Instruction opcode and operand information
  // -- Pipelines and scheduling information
  // -- Stack frame information
  // -- Selection DAG lowering information
  //
  virtual const TargetInstrInfo        *getInstrInfo() const { return 0; }
  virtual const TargetFrameInfo        *getFrameInfo() const { return 0; }
  virtual       TargetLowering    *getTargetLowering() const { return 0; }
  virtual const TargetData            *getTargetData() const { return 0; }
  
  
  /// getTargetAsmInfo - Return target specific asm information.
  ///
  const TargetAsmInfo *getTargetAsmInfo() const {
    if (!AsmInfo) AsmInfo = createTargetAsmInfo();
    return AsmInfo;
  }
  
  /// getSubtarget - This method returns a pointer to the specified type of
  /// TargetSubtarget.  In debug builds, it verifies that the object being
  /// returned is of the correct type.
  template<typename STC> const STC &getSubtarget() const {
    const TargetSubtarget *TST = getSubtargetImpl();
    assert(TST && dynamic_cast<const STC*>(TST) &&
           "Not the right kind of subtarget!");
    return *static_cast<const STC*>(TST);
  }

  /// getRegisterInfo - If register information is available, return it.  If
  /// not, return null.  This is kept separate from RegInfo until RegInfo has
  /// details of graph coloring register allocation removed from it.
  ///
  virtual const TargetRegisterInfo *getRegisterInfo() const { return 0; }

  /// getJITInfo - If this target supports a JIT, return information for it,
  /// otherwise return null.
  ///
  virtual TargetJITInfo *getJITInfo() { return 0; }
  
  /// getInstrItineraryData - Returns instruction itinerary data for the target
  /// or specific subtarget.
  ///
  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItineraryData();
  }

  /// getMachOWriterInfo - If this target supports a Mach-O writer, return
  /// information for it, otherwise return null.
  /// 
  virtual const TargetMachOWriterInfo *getMachOWriterInfo() const { return 0; }

  /// getELFWriterInfo - If this target supports an ELF writer, return
  /// information for it, otherwise return null.
  /// 
  virtual const TargetELFWriterInfo *getELFWriterInfo() const { return 0; }

  /// getRelocationModel - Returns the code generation relocation model. The
  /// choices are static, PIC, and dynamic-no-pic, and target default.
  static Reloc::Model getRelocationModel();

  /// setRelocationModel - Sets the code generation relocation model.
  static void setRelocationModel(Reloc::Model Model);

  /// getCodeModel - Returns the code model. The choices are small, kernel,
  /// medium, large, and target default.
  static CodeModel::Model getCodeModel();

  /// setCodeModel - Sets the code model.
  static void setCodeModel(CodeModel::Model Model);

  /// CodeGenFileType - These enums are meant to be passed into
  /// addPassesToEmitFile to indicate what type of file to emit.
  enum CodeGenFileType {
    AssemblyFile, ObjectFile, DynamicLibrary
  };

  /// getEnableTailMergeDefault - the default setting for -enable-tail-merge
  /// on this target.  User flag overrides.
  virtual bool getEnableTailMergeDefault() const { return true; }

  /// addPassesToEmitFile - Add passes to the specified pass manager to get the
  /// specified file emitted.  Typically this will involve several steps of code
  /// generation.  If Fast is set to true, the code generator should emit code
  /// as fast as possible, though the generated code may be less efficient.
  /// This method should return FileModel::Error if emission of this file type
  /// is not supported.
  ///
  virtual FileModel::Model addPassesToEmitFile(PassManagerBase &,
                                               raw_ostream &,
                                               CodeGenFileType,
                                               bool /*Fast*/) {
    return FileModel::None;
  }

  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &,
                                         MachineCodeEmitter *, bool /*Fast*/) {
    return true;
  }
 
  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &,
                                          MachineCodeEmitter &,
                                          bool /*Fast*/) {
    return true;
  }

  /// addPassesToEmitWholeFile - This method can be implemented by targets that 
  /// require having the entire module at once.  This is not recommended, do not
  /// use this.
  virtual bool WantsWholeFile() const { return false; }
  virtual bool addPassesToEmitWholeFile(PassManager &, raw_ostream &,
                                        CodeGenFileType, bool /*Fast*/) {
    return true;
  }
};

/// LLVMTargetMachine - This class describes a target machine that is
/// implemented with the LLVM target-independent code generator.
///
class LLVMTargetMachine : public TargetMachine {
protected: // Can only create subclasses.
  LLVMTargetMachine() { }

  /// addCommonCodeGenPasses - Add standard LLVM codegen passes used for
  /// both emitting to assembly files or machine code output.
  ///
  bool addCommonCodeGenPasses(PassManagerBase &, bool /*Fast*/);

public:
  
  /// addPassesToEmitFile - Add passes to the specified pass manager to get the
  /// specified file emitted.  Typically this will involve several steps of code
  /// generation.  If Fast is set to true, the code generator should emit code
  /// as fast as possible, though the generated code may be less efficient.
  /// This method should return FileModel::Error if emission of this file type
  /// is not supported.
  ///
  /// The default implementation of this method adds components from the
  /// LLVM retargetable code generator, invoking the methods below to get
  /// target-specific passes in standard locations.
  ///
  virtual FileModel::Model addPassesToEmitFile(PassManagerBase &PM,
                                               raw_ostream &Out,
                                               CodeGenFileType FileType,
                                               bool Fast);
  
  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &PM,
                                         MachineCodeEmitter *MCE, bool Fast);
 
  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &PM,
                                          MachineCodeEmitter &MCE, bool Fast);
  
  /// Target-Independent Code Generator Pass Configuration Options.
  
  /// addInstSelector - This method should add any "last minute" LLVM->LLVM
  /// passes, then install an instruction selector pass, which converts from
  /// LLVM code to machine instructions.
  virtual bool addInstSelector(PassManagerBase &, bool /*Fast*/) {
    return true;
  }

  /// addPreRegAllocPasses - This method may be implemented by targets that want
  /// to run passes immediately before register allocation. This should return
  /// true if -print-machineinstrs should print after these passes.
  virtual bool addPreRegAlloc(PassManagerBase &, bool /*Fast*/) {
    return false;
  }

  /// addPostRegAllocPasses - This method may be implemented by targets that
  /// want to run passes after register allocation but before prolog-epilog
  /// insertion.  This should return true if -print-machineinstrs should print
  /// after these passes.
  virtual bool addPostRegAlloc(PassManagerBase &, bool /*Fast*/) {
    return false;
  }
  
  /// addPreEmitPass - This pass may be implemented by targets that want to run
  /// passes immediately before machine code is emitted.  This should return
  /// true if -print-machineinstrs should print out the code after the passes.
  virtual bool addPreEmitPass(PassManagerBase &, bool /*Fast*/) {
    return false;
  }
  
  
  /// addAssemblyEmitter - This pass should be overridden by the target to add
  /// the asmprinter, if asm emission is supported.  If this is not supported,
  /// 'true' should be returned.
  virtual bool addAssemblyEmitter(PassManagerBase &, bool /*Fast*/, 
                                  raw_ostream &) {
    return true;
  }
  
  /// addCodeEmitter - This pass should be overridden by the target to add a
  /// code emitter, if supported.  If this is not supported, 'true' should be
  /// returned. If DumpAsm is true, the generated assembly is printed to cerr.
  virtual bool addCodeEmitter(PassManagerBase &, bool /*Fast*/,
                              bool /*DumpAsm*/, MachineCodeEmitter &) {
    return true;
  }

  /// addSimpleCodeEmitter - This pass should be overridden by the target to add
  /// a code emitter (without setting flags), if supported.  If this is not
  /// supported, 'true' should be returned.  If DumpAsm is true, the generated
  /// assembly is printed to cerr.
  virtual bool addSimpleCodeEmitter(PassManagerBase &, bool /*Fast*/,
                                    bool /*DumpAsm*/, MachineCodeEmitter &) {
    return true;
  }

  /// getEnableTailMergeDefault - the default setting for -enable-tail-merge
  /// on this target.  User flag overrides.
  virtual bool getEnableTailMergeDefault() const { return true; }
};

} // End llvm namespace

#endif
