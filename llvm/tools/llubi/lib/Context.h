//===--- Context.h - State Tracking for llubi -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLUBI_CONTEXT_H
#define LLVM_TOOLS_LLUBI_CONTEXT_H

#include "Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Module.h"
#include <map>
#include <random>

namespace llvm::ubi {

enum class MemInitKind {
  Zeroed,
  Uninitialized,
  Poisoned,
};

enum class MemoryObjectState {
  // This memory object is accessible.
  // Valid transitions:
  //   -> Dead (after the end of lifetime of an alloca)
  //   -> Freed (after free is called on a heap object)
  Alive,
  // This memory object is out of lifetime. It is OK to perform
  // operations that do not access its content, e.g., getelementptr.
  // Otherwise, an immediate UB occurs.
  // Valid transition:
  //   -> Alive (after the start of lifetime of an alloca)
  Dead,
  // This heap memory object has been freed. Any access to it
  // causes immediate UB. Like dead objects, it is still possible to
  // perform operations that do not access its content.
  Freed,
};

enum class UndefValueBehavior {
  NonDeterministic, // Each use of the undef value can yield different results.
  Zero,             // All uses of the undef value yield zero.
};

class MemoryObject : public RefCountedBase<MemoryObject> {
  uint64_t Address;
  uint64_t Size;
  SmallVector<Byte, 8> Bytes;
  StringRef Name;
  unsigned AS;

  MemoryObjectState State;
  bool IsConstant = false;

public:
  MemoryObject(uint64_t Addr, uint64_t Size, StringRef Name, unsigned AS,
               MemInitKind InitKind);
  MemoryObject(const MemoryObject &) = delete;
  MemoryObject(MemoryObject &&) = delete;
  MemoryObject &operator=(const MemoryObject &) = delete;
  MemoryObject &operator=(MemoryObject &&) = delete;
  ~MemoryObject();

  uint64_t getAddress() const { return Address; }
  uint64_t getSize() const { return Size; }
  StringRef getName() const { return Name; }
  unsigned getAddressSpace() const { return AS; }
  MemoryObjectState getState() const { return State; }
  void setState(MemoryObjectState S) { State = S; }
  bool isConstant() const { return IsConstant; }
  void setIsConstant(bool C) { IsConstant = C; }

  bool inBounds(const APInt &NewAddr) const {
    return NewAddr.uge(Address) && NewAddr.ule(Address + Size);
  }

  Byte &operator[](uint64_t Offset) {
    assert(Offset < Size && "Offset out of bounds");
    return Bytes[Offset];
  }
  ArrayRef<Byte> getBytes() const { return Bytes; }
  MutableArrayRef<Byte> getBytes() { return Bytes; }

  void markAsFreed();
};

/// An interface for handling events and managing outputs during interpretation.
/// If the handler returns false from any of the methods, the interpreter will
/// stop execution immediately.
class EventHandler {
public:
  virtual ~EventHandler() = default;

  virtual bool onInstructionExecuted(Instruction &I, const AnyValue &Result) {
    return true;
  }
  virtual void onError(StringRef Msg) {}
  virtual void onUnrecognizedInstruction(Instruction &I) {}
  virtual void onImmediateUB(StringRef Msg) {}
  virtual bool onBBJump(Instruction &I, BasicBlock &To) { return true; }
  virtual bool onFunctionEntry(Function &F, ArrayRef<AnyValue> Args,
                               CallBase *CallSite) {
    return true;
  }
  virtual bool onFunctionExit(Function &F, const AnyValue &RetVal) {
    return true;
  }
  virtual bool onPrint(StringRef Msg) {
    outs() << Msg;
    return true;
  }
};

/// The global context for the interpreter.
/// It tracks global state such as heap memory objects and floating point
/// environment.
class Context {
  // Module
  LLVMContext &Ctx;
  Module &M;
  const DataLayout &DL;
  const TargetLibraryInfoImpl TLIImpl;

  // Configuration
  uint64_t MaxMem = 0;
  uint32_t VScale = 4;
  uint32_t MaxSteps = 0;
  uint32_t MaxStackDepth = 256;
  UndefValueBehavior UndefBehavior = UndefValueBehavior::NonDeterministic;

  std::mt19937_64 Rng;

  // Memory
  uint64_t UsedMem = 0;
  // The addresses of memory objects are monotonically increasing.
  // For now we don't model the behavior of address reuse, which is common
  // with stack coloring.
  uint64_t AllocationBase = 8;
  // Maintains a global list of 'exposed' provenances. This is used to form a
  // pointer with an exposed provenance.
  // FIXME: Currently all the allocations are considered exposed, regardless of
  // their interaction with ptrtoint. That is, ptrtoint is allowed to recover
  // the provenance of any allocation. We may track the exposed provenances more
  // precisely after we make ptrtoint have the implicit side-effect of exposing
  // the provenance.
  std::map<uint64_t, IntrusiveRefCntPtr<MemoryObject>> MemoryObjects;
  AnyValue fromBytes(ArrayRef<Byte> Bytes, Type *Ty, uint32_t &OffsetInBits,
                     bool CheckPaddingBits);
  void toBytes(const AnyValue &Val, Type *Ty, uint32_t &OffsetInBits,
               MutableArrayRef<Byte> Bytes, bool PaddingBits);

  // Constants
  // Use std::map to avoid iterator/reference invalidation.
  std::map<Constant *, AnyValue> ConstCache;
  DenseMap<Function *, Pointer> FuncAddrMap;
  DenseMap<BasicBlock *, Pointer> BlockAddrMap;
  DenseMap<uint64_t, std::pair<Function *, IntrusiveRefCntPtr<MemoryObject>>>
      ValidFuncTargets;
  DenseMap<uint64_t, std::pair<BasicBlock *, IntrusiveRefCntPtr<MemoryObject>>>
      ValidBlockTargets;
  AnyValue getConstantValueImpl(Constant *C);

  // TODO: errno and fpenv

public:
  explicit Context(Module &M);
  Context(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(const Context &) = delete;
  Context &operator=(Context &&) = delete;
  ~Context();

  void setMemoryLimit(uint64_t Max) { MaxMem = Max; }
  void setVScale(uint32_t VS) { VScale = VS; }
  void setMaxSteps(uint32_t MS) { MaxSteps = MS; }
  void setMaxStackDepth(uint32_t Depth) { MaxStackDepth = Depth; }
  uint64_t getMemoryLimit() const { return MaxMem; }
  uint32_t getVScale() const { return VScale; }
  uint32_t getMaxSteps() const { return MaxSteps; }
  uint32_t getMaxStackDepth() const { return MaxStackDepth; }
  void setUndefValueBehavior(UndefValueBehavior UB) { UndefBehavior = UB; }
  void reseed(uint32_t Seed) { Rng.seed(Seed); }

  LLVMContext &getContext() const { return Ctx; }
  const DataLayout &getDataLayout() const { return DL; }
  const TargetLibraryInfoImpl &getTLIImpl() const { return TLIImpl; }
  uint32_t getEVL(ElementCount EC) const {
    if (EC.isScalable())
      return VScale * EC.getKnownMinValue();
    return EC.getFixedValue();
  }
  uint64_t getEffectiveTypeAllocSize(Type *Ty);
  uint64_t getEffectiveTypeStoreSize(Type *Ty);

  const AnyValue &getConstantValue(Constant *C);
  IntrusiveRefCntPtr<MemoryObject> allocate(uint64_t Size, uint64_t Align,
                                            StringRef Name, unsigned AS,
                                            MemInitKind InitKind);
  bool free(uint64_t Address);
  /// Derive a pointer from a memory object with offset 0.
  /// Please use Pointer's interface for further manipulations.
  Pointer deriveFromMemoryObject(IntrusiveRefCntPtr<MemoryObject> Obj);
  /// Convert byte sequence to an value of the given type. Uninitialized bits
  /// are flushed according to the options.
  AnyValue fromBytes(ArrayRef<Byte> Bytes, Type *Ty);
  /// Convert a value to byte sequence. Padding bits are set to zero.
  void toBytes(const AnyValue &Val, Type *Ty, MutableArrayRef<Byte> Bytes);
  /// Direct memory load without checks.
  AnyValue load(MemoryObject &MO, uint64_t Offset, Type *ValTy);
  /// Direct memory store without checks.
  void store(MemoryObject &MO, uint64_t Offset, const AnyValue &Val,
             Type *ValTy);
  void storeRawBytes(MemoryObject &MO, uint64_t Offset, const void *Data,
                     uint64_t Size);

  Function *getTargetFunction(const Pointer &Ptr);
  BasicBlock *getTargetBlock(const Pointer &Ptr);

  /// Initialize global variables and function/block objects. This function
  /// should be called before executing any function. Returns false if the
  /// initialization fails (e.g., the memory limit is exceeded during
  /// initialization).
  bool initGlobalValues();
  /// Execute the function \p F with arguments \p Args, and store the return
  /// value in \p RetVal if the function is not void.
  /// Returns true if the function executed successfully. False indicates an
  /// error occurred during execution.
  bool runFunction(Function &F, ArrayRef<AnyValue> Args, AnyValue &RetVal,
                   EventHandler &Handler);
};

} // namespace llvm::ubi

#endif
