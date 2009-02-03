//===- ARMJITInfo.h - ARM implementation of the JIT interface  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMJITINFO_H
#define ARMJITINFO_H

#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetJITInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
  class ARMTargetMachine;

  class ARMJITInfo : public TargetJITInfo {
    // ConstPoolId2AddrMap - A map from constant pool ids to the corresponding
    // CONSTPOOL_ENTRY addresses.
    SmallVector<intptr_t, 16> ConstPoolId2AddrMap;

    // JumpTableId2AddrMap - A map from inline jumptable ids to the
    // corresponding inline jump table bases.
    SmallVector<intptr_t, 16> JumpTableId2AddrMap;

    // PCLabelMap - A map from PC labels to addresses.
    DenseMap<unsigned, intptr_t> PCLabelMap;

    // Sym2IndirectSymMap - A map from symbol (GlobalValue and ExternalSymbol)
    // addresses to their indirect symbol addresses.
    DenseMap<void*, intptr_t> Sym2IndirectSymMap;

    // IsPIC - True if the relocation model is PIC. This is used to determine
    // how to codegen function stubs.
    bool IsPIC;

  public:
    explicit ARMJITInfo() : IsPIC(false) { useGOT = false; }

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);

    /// emitGlobalValueIndirectSym - Use the specified MachineCodeEmitter object
    /// to emit an indirect symbol which contains the address of the specified
    /// ptr.
    virtual void *emitGlobalValueIndirectSym(const GlobalValue* GV, void *ptr,
                                            MachineCodeEmitter &MCE);

    /// emitFunctionStub - Use the specified MachineCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.
    virtual void *emitFunctionStub(const Function* F, void *Fn,
                                   MachineCodeEmitter &MCE);

    /// getLazyResolverFunction - Expose the lazy resolver to the JIT.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);

    /// hasCustomConstantPool - Allows a target to specify that constant
    /// pool address resolution is handled by the target.
    virtual bool hasCustomConstantPool() const { return true; }

    /// hasCustomJumpTables - Allows a target to specify that jumptables
    /// are emitted by the target.
    virtual bool hasCustomJumpTables() const { return true; }

    /// allocateSeparateGVMemory - If true, globals should be placed in
    /// separately allocated heap memory rather than in the same
    /// code memory allocated by MachineCodeEmitter.
    virtual bool allocateSeparateGVMemory() const {
#ifdef __APPLE__
      return true;
#else
      return false;
#endif
    }

    /// Initialize - Initialize internal stage for the function being JITted.
    /// Resize constant pool ids to CONSTPOOL_ENTRY addresses map; resize
    /// jump table ids to jump table bases map; remember if codegen relocation
    /// model is PIC.
    void Initialize(const MachineFunction &MF, bool isPIC) {
      const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
      ConstPoolId2AddrMap.resize(AFI->getNumConstPoolEntries());
      JumpTableId2AddrMap.resize(AFI->getNumJumpTables());
      IsPIC = isPIC;
    }

    /// getConstantPoolEntryAddr - The ARM target puts all constant
    /// pool entries into constant islands. This returns the address of the
    /// constant pool entry of the specified index.
    intptr_t getConstantPoolEntryAddr(unsigned CPI) const {
      assert(CPI < ConstPoolId2AddrMap.size());
      return ConstPoolId2AddrMap[CPI];
    }

    /// addConstantPoolEntryAddr - Map a Constant Pool Index to the address
    /// where its associated value is stored. When relocations are processed,
    /// this value will be used to resolve references to the constant.
    void addConstantPoolEntryAddr(unsigned CPI, intptr_t Addr) {
      assert(CPI < ConstPoolId2AddrMap.size());
      ConstPoolId2AddrMap[CPI] = Addr;
    }

    /// getJumpTableBaseAddr - The ARM target inline all jump tables within
    /// text section of the function. This returns the address of the base of
    /// the jump table of the specified index.
    intptr_t getJumpTableBaseAddr(unsigned JTI) const {
      assert(JTI < JumpTableId2AddrMap.size());
      return JumpTableId2AddrMap[JTI];
    }

    /// addJumpTableBaseAddr - Map a jump table index to the address where
    /// the corresponding inline jump table is emitted. When relocations are
    /// processed, this value will be used to resolve references to the
    /// jump table.
    void addJumpTableBaseAddr(unsigned JTI, intptr_t Addr) {
      assert(JTI < JumpTableId2AddrMap.size());
      JumpTableId2AddrMap[JTI] = Addr;
    }

    /// getPCLabelAddr - Retrieve the address of the PC label of the specified id.
    intptr_t getPCLabelAddr(unsigned Id) const {
      DenseMap<unsigned, intptr_t>::const_iterator I = PCLabelMap.find(Id);
      assert(I != PCLabelMap.end());
      return I->second;
    }

    /// addPCLabelAddr - Remember the address of the specified PC label.
    void addPCLabelAddr(unsigned Id, intptr_t Addr) {
      PCLabelMap.insert(std::make_pair(Id, Addr));
    }

    /// getIndirectSymAddr - Retrieve the address of the indirect symbol of the
    /// specified symbol located at address. Returns 0 if the indirect symbol
    /// has not been emitted.
    intptr_t getIndirectSymAddr(void *Addr) const {
      DenseMap<void*,intptr_t>::const_iterator I= Sym2IndirectSymMap.find(Addr);
      if (I != Sym2IndirectSymMap.end())
        return I->second;
      return 0;
    }

    /// addIndirectSymAddr - Add a mapping from address of an emitted symbol to
    /// its indirect symbol address.
    void addIndirectSymAddr(void *SymAddr, intptr_t IndSymAddr) {
      Sym2IndirectSymMap.insert(std::make_pair(SymAddr, IndSymAddr));
    }

  private:
    /// resolveRelocDestAddr - Resolve the resulting address of the relocation
    /// if it's not already solved. Constantpool entries must be resolved by
    /// ARM target.
    intptr_t resolveRelocDestAddr(MachineRelocation *MR) const;
  };
}

#endif
