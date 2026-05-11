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
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/Module.h"
#include <map>
#include <random>

namespace llvm::ubi {

enum class MemInitKind {
  Zeroed,
  Uninitialized,
  Poisoned,
};

enum class MemAllocKind {
  Global,
  Stack,
  Malloc,
  New,
  NewArray,
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

enum class NaNPropagationBehavior {
  NonDeterministic, // Non-deterministically choose from valid NaN results
  PreferredNaN,     // The quiet bit is set and the payload is all-zero
  QuietingNaN,  // The quiet bit is set and the payload is copied from any input
                // operand that is a NaN
  UnchangedNaN, // The quiet bit and payload are copied from any input operand
                // that is a NaN
  TargetSpecificNaN // The quiet bit is set and the payload is picked from a
                    // known target-specific set of "extra" possible NaN
                    // payloads
};

struct ProgramExitInfo {
  enum class ProgramExitKind {
    // Program exited via a normal return
    Returned,
    // Program exited with an interpreter error (UB/Unsupported
    // instruction/etc.)
    Failed,
    // Program exited via a call to exit()
    Exited,
    // Program exited via a call to abort()
    Aborted,
    // Program exited via a call to terminate()
    Terminated,
  };

  ProgramExitKind Kind;
  uint64_t ExitCode;

  explicit ProgramExitInfo(ProgramExitKind Kind, uint64_t ExitCode)
      : Kind(Kind), ExitCode(ExitCode) {}

  bool isExitedByLibcall() const {
    return Kind == ProgramExitKind::Exited ||
           Kind == ProgramExitKind::Aborted ||
           Kind == ProgramExitKind::Terminated;
  }
};

class MemoryObject : public RefCountedBase<MemoryObject> {
  uint64_t Address;
  uint64_t Size;
  SmallVector<Byte, 8> Bytes;
  StringRef Name;
  unsigned AS;

  MemoryObjectState State;
  MemAllocKind AllocKind;
  bool IsConstant = false;

public:
  MemoryObject(uint64_t Addr, uint64_t Size, StringRef Name, unsigned AS,
               MemInitKind InitKind, MemAllocKind AllocKind);
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
  MemAllocKind getAllocKind() const { return AllocKind; }
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

  bool isGlobal() const;
  bool isStackAllocated() const;
  bool isHeapAllocated() const;
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
  virtual void onProgramExit(const ProgramExitInfo &ExitInfo) {}
  virtual bool onPrint(StringRef Msg) {
    outs() << Msg;
    outs().flush();
    return true;
  }
};

/// Endianness aware accessor for bytes.
template <typename ArrayRefT> class BytesView {
  ArrayRefT Bytes;
  bool IsLittleEndian;

public:
  explicit BytesView(ArrayRefT Ref, const DataLayout &DL)
      : Bytes(Ref), IsLittleEndian(DL.isLittleEndian()) {}

  auto &operator[](uint32_t Index) {
    return Bytes[IsLittleEndian ? Index : Bytes.size() - 1 - Index];
  }
};

using ConstBytesView = BytesView<ArrayRef<Byte>>;
using MutableBytesView = BytesView<MutableArrayRef<Byte>>;

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
  bool Deterministic = false;
  UndefValueBehavior UndefBehavior = UndefValueBehavior::NonDeterministic;
  NaNPropagationBehavior NaNBehavior = NaNPropagationBehavior::NonDeterministic;
  bool FusedMultiplyAdd = false;

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
  AnyValue fromBytes(ConstBytesView Bytes, Type *Ty, uint32_t OffsetInBits,
                     bool CheckPaddingBits, bool *ContainsUndefinedBits);
  void toBytes(const AnyValue &Val, Type *Ty, uint32_t OffsetInBits,
               MutableBytesView Bytes, bool PaddingBits);

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

  // Floating-point environment
  RoundingMode CurrentRoundingMode = RoundingMode::NearestTiesToEven;
  fp::ExceptionBehavior CurrentExceptionBehavior =
      fp::ExceptionBehavior::ebIgnore;

  // TODO: errno

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
  void setFusedMultiplyAdd(bool F) { FusedMultiplyAdd = F; }
  uint64_t getMemoryLimit() const { return MaxMem; }
  uint32_t getVScale() const { return VScale; }
  uint32_t getMaxSteps() const { return MaxSteps; }
  uint32_t getMaxStackDepth() const { return MaxStackDepth; }
  void setDeterministic(bool D) { Deterministic = D; }
  bool isDeterministic() const { return Deterministic; }
  bool mayUseNonDeterminism() const { return !Deterministic; }
  UndefValueBehavior getEffectiveUndefValueBehavior() const;
  NaNPropagationBehavior getEffectiveNaNPropagationBehavior() const;
  bool fuseMultiplyAdd() const { return FusedMultiplyAdd; }
  void setUndefValueBehavior(UndefValueBehavior UB) { UndefBehavior = UB; }
  void setNaNPropagationBehavior(NaNPropagationBehavior NaNBehav) {
    NaNBehavior = NaNBehav;
  }
  void reseed(uint32_t Seed) { Rng.seed(Seed); }

  LLVMContext &getContext() const { return Ctx; }
  const DataLayout &getDataLayout() const { return DL; }
  const Triple &getTargetTriple() const { return M.getTargetTriple(); }
  const TargetLibraryInfoImpl &getTLIImpl() const { return TLIImpl; }
  /// Get the effective vector length for a vector type.
  uint32_t getEVL(ElementCount EC) const {
    if (EC.isScalable())
      return VScale * EC.getKnownMinValue();
    return EC.getFixedValue();
  }
  /// The result is multiplied by VScale for scalable type sizes.
  uint64_t getEffectiveTypeSize(TypeSize Size) const {
    if (Size.isScalable())
      return VScale * Size.getKnownMinValue();
    return Size.getFixedValue();
  }
  /// Returns DL.getTypeAllocSize/getTypeStoreSize for the given type.
  /// An exception to this is that for scalable vector types, the size is
  /// computed as if the vector has getEVL(ElementCount) elements.
  uint64_t getEffectiveTypeAllocSize(Type *Ty);
  uint64_t getEffectiveTypeStoreSize(Type *Ty);

  const AnyValue &getConstantValue(Constant *C);
  IntrusiveRefCntPtr<MemoryObject> allocate(uint64_t Size, uint64_t Align,
                                            StringRef Name, unsigned AS,
                                            MemInitKind InitKind,
                                            MemAllocKind AllocKind);
  bool free(const MemoryObject &Obj);
  /// Derive a pointer from a memory object with offset 0.
  /// Please use Pointer's interface for further manipulations.
  Pointer deriveFromMemoryObject(IntrusiveRefCntPtr<MemoryObject> Obj);
  /// Convert byte sequence to a value of the given type. Uninitialized bits are
  /// flushed according to the options.
  /// If \p ContainsUndefinedBits is provided, it will be set to true when there
  /// are poison or undef bits in the value (i.e., padding bits are ignored).
  AnyValue fromBytes(ArrayRef<Byte> Bytes, Type *Ty,
                     bool *ContainsUndefinedBits = nullptr);
  /// Convert a value to byte sequence. Padding bits are set to zero.
  void toBytes(const AnyValue &Val, Type *Ty, MutableArrayRef<Byte> Bytes);
  /// Direct memory load without checks.
  AnyValue load(MemoryObject &MO, uint64_t Offset, Type *ValTy,
                bool *ContainsUndefinedBits = nullptr);
  /// Direct memory store without checks.
  void store(MemoryObject &MO, uint64_t Offset, const AnyValue &Val,
             Type *ValTy);
  void storeRawBytes(MemoryObject &MO, uint64_t Offset, const void *Data,
                     uint64_t Size);

  /// Freeze the value in-place.
  void freeze(AnyValue &Val, Type *Ty);

  Function *getTargetFunction(const Pointer &Ptr);
  BasicBlock *getTargetBlock(const Pointer &Ptr);

  /// Initialize global variables and function/block objects. This function
  /// should be called before executing any function. Returns false if the
  /// initialization fails (e.g., the memory limit is exceeded during
  /// initialization).
  bool initGlobalValues();
  /// Execute the function \p F with arguments \p Args, and store the return
  /// value in \p RetVal if the function is not void.
  /// Returns a `ProgramExitInfo` indicating how the program finished:
  /// Kind = Returned: The program executed successfully and returned normally.
  /// Kind = Failed: The interpreter encountered an error and could not execute
  /// the program.
  /// Kind = Exited/Aborted/Terminated: The program ended via an
  /// explicit call to `exit()`, `abort()`, or `terminate()`.
  ProgramExitInfo runFunction(Function &F, ArrayRef<AnyValue> Args,
                              AnyValue &RetVal, EventHandler &Handler);

  RoundingMode getCurrentRoundingMode() const;
  fp::ExceptionBehavior getCurrentExceptionBehavior() const;
  void setCurrentRoundingMode(RoundingMode RM);
  void setCurrentExceptionBehavior(fp::ExceptionBehavior EB);
  bool isDefaultFPEnv() const;

  bool getRandomBool();
  uint64_t getRandomUInt64();
};

} // namespace llvm::ubi

#endif
