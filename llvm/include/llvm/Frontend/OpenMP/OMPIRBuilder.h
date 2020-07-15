//===- IR/OpenMPIRBuilder.h - OpenMP encoding builder for LLVM IR - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpenMPIRBuilder class and helpers used as a convenient
// way to create LLVM instructions for OpenMP directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_IR_IRBUILDER_H
#define LLVM_OPENMP_IR_IRBUILDER_H

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

/// An interface to create LLVM-IR for OpenMP directives.
///
/// Each OpenMP directive has a corresponding public generator method.
class OpenMPIRBuilder {
public:
  /// Create a new OpenMPIRBuilder operating on the given module \p M. This will
  /// not have an effect on \p M (see initialize).
  OpenMPIRBuilder(Module &M) : M(M), Builder(M.getContext()) {}

  /// Initialize the internal state, this will put structures types and
  /// potentially other helpers into the underlying module. Must be called
  /// before any other method and only once!
  void initialize();

  /// Finalize the underlying module, e.g., by outlining regions.
  void finalize();

  /// Add attributes known for \p FnID to \p Fn.
  void addAttributes(omp::RuntimeFunction FnID, Function &Fn);

  /// Type used throughout for insertion points.
  using InsertPointTy = IRBuilder<>::InsertPoint;

  /// Callback type for variable finalization (think destructors).
  ///
  /// \param CodeGenIP is the insertion point at which the finalization code
  ///                  should be placed.
  ///
  /// A finalize callback knows about all objects that need finalization, e.g.
  /// destruction, when the scope of the currently generated construct is left
  /// at the time, and location, the callback is invoked.
  using FinalizeCallbackTy = std::function<void(InsertPointTy CodeGenIP)>;

  struct FinalizationInfo {
    /// The finalization callback provided by the last in-flight invocation of
    /// CreateXXXX for the directive of kind DK.
    FinalizeCallbackTy FiniCB;

    /// The directive kind of the innermost directive that has an associated
    /// region which might require finalization when it is left.
    omp::Directive DK;

    /// Flag to indicate if the directive is cancellable.
    bool IsCancellable;
  };

  /// Push a finalization callback on the finalization stack.
  ///
  /// NOTE: Temporary solution until Clang CG is gone.
  void pushFinalizationCB(const FinalizationInfo &FI) {
    FinalizationStack.push_back(FI);
  }

  /// Pop the last finalization callback from the finalization stack.
  ///
  /// NOTE: Temporary solution until Clang CG is gone.
  void popFinalizationCB() { FinalizationStack.pop_back(); }

  /// Callback type for body (=inner region) code generation
  ///
  /// The callback takes code locations as arguments, each describing a
  /// location at which code might need to be generated or a location that is
  /// the target of control transfer.
  ///
  /// \param AllocaIP is the insertion point at which new alloca instructions
  ///                 should be placed.
  /// \param CodeGenIP is the insertion point at which the body code should be
  ///                  placed.
  /// \param ContinuationBB is the basic block target to leave the body.
  ///
  /// Note that all blocks pointed to by the arguments have terminators.
  using BodyGenCallbackTy =
      function_ref<void(InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                        BasicBlock &ContinuationBB)>;

  /// Callback type for variable privatization (think copy & default
  /// constructor).
  ///
  /// \param AllocaIP is the insertion point at which new alloca instructions
  ///                 should be placed.
  /// \param CodeGenIP is the insertion point at which the privatization code
  ///                  should be placed.
  /// \param Val The value beeing copied/created.
  /// \param ReplVal The replacement value, thus a copy or new created version
  ///                of \p Val.
  ///
  /// \returns The new insertion point where code generation continues and
  ///          \p ReplVal the replacement of \p Val.
  using PrivatizeCallbackTy = function_ref<InsertPointTy(
      InsertPointTy AllocaIP, InsertPointTy CodeGenIP, Value &Val,
      Value *&ReplVal)>;

  /// Description of a LLVM-IR insertion point (IP) and a debug/source location
  /// (filename, line, column, ...).
  struct LocationDescription {
    template <typename T, typename U>
    LocationDescription(const IRBuilder<T, U> &IRB)
        : IP(IRB.saveIP()), DL(IRB.getCurrentDebugLocation()) {}
    LocationDescription(const InsertPointTy &IP) : IP(IP) {}
    LocationDescription(const InsertPointTy &IP, const DebugLoc &DL)
        : IP(IP), DL(DL) {}
    InsertPointTy IP;
    DebugLoc DL;
  };

  /// Emitter methods for OpenMP directives.
  ///
  ///{

  /// Generator for '#omp barrier'
  ///
  /// \param Loc The location where the barrier directive was encountered.
  /// \param DK The kind of directive that caused the barrier.
  /// \param ForceSimpleCall Flag to force a simple (=non-cancellation) barrier.
  /// \param CheckCancelFlag Flag to indicate a cancel barrier return value
  ///                        should be checked and acted upon.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy CreateBarrier(const LocationDescription &Loc, omp::Directive DK,
                              bool ForceSimpleCall = false,
                              bool CheckCancelFlag = true);

  /// Generator for '#omp cancel'
  ///
  /// \param Loc The location where the directive was encountered.
  /// \param IfCondition The evaluated 'if' clause expression, if any.
  /// \param CanceledDirective The kind of directive that is cancled.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy CreateCancel(const LocationDescription &Loc, Value *IfCondition,
                             omp::Directive CanceledDirective);

  /// Generator for '#omp parallel'
  ///
  /// \param Loc The insert and source location description.
  /// \param BodyGenCB Callback that will generate the region code.
  /// \param PrivCB Callback to copy a given variable (think copy constructor).
  /// \param FiniCB Callback to finalize variable copies.
  /// \param IfCondition The evaluated 'if' clause expression, if any.
  /// \param NumThreads The evaluated 'num_threads' clause expression, if any.
  /// \param ProcBind The value of the 'proc_bind' clause (see ProcBindKind).
  /// \param IsCancellable Flag to indicate a cancellable parallel region.
  ///
  /// \returns The insertion position *after* the parallel.
  IRBuilder<>::InsertPoint
  CreateParallel(const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
                 PrivatizeCallbackTy PrivCB, FinalizeCallbackTy FiniCB,
                 Value *IfCondition, Value *NumThreads,
                 omp::ProcBindKind ProcBind, bool IsCancellable);

  /// Generator for '#omp flush'
  ///
  /// \param Loc The location where the flush directive was encountered
  void CreateFlush(const LocationDescription &Loc);

  /// Generator for '#omp taskwait'
  ///
  /// \param Loc The location where the taskwait directive was encountered.
  void CreateTaskwait(const LocationDescription &Loc);

  /// Generator for '#omp taskyield'
  ///
  /// \param Loc The location where the taskyield directive was encountered.
  void CreateTaskyield(const LocationDescription &Loc);

  ///}

  /// Return the insertion point used by the underlying IRBuilder.
  InsertPointTy getInsertionPoint() { return Builder.saveIP(); }

  /// Update the internal location to \p Loc.
  bool updateToLocation(const LocationDescription &Loc) {
    Builder.restoreIP(Loc.IP);
    Builder.SetCurrentDebugLocation(Loc.DL);
    return Loc.IP.getBlock() != nullptr;
  }

  /// Return the function declaration for the runtime function with \p FnID.
  FunctionCallee getOrCreateRuntimeFunction(Module &M,
                                            omp::RuntimeFunction FnID);

  Function *getOrCreateRuntimeFunctionPtr(omp::RuntimeFunction FnID);

  /// Return the (LLVM-IR) string describing the source location \p LocStr.
  Constant *getOrCreateSrcLocStr(StringRef LocStr);

  /// Return the (LLVM-IR) string describing the default source location.
  Constant *getOrCreateDefaultSrcLocStr();

  /// Return the (LLVM-IR) string describing the source location \p Loc.
  Constant *getOrCreateSrcLocStr(const LocationDescription &Loc);

  /// Return an ident_t* encoding the source location \p SrcLocStr and \p Flags.
  Value *getOrCreateIdent(Constant *SrcLocStr,
                          omp::IdentFlag Flags = omp::IdentFlag(0));

  /// Generate control flow and cleanup for cancellation.
  ///
  /// \param CancelFlag Flag indicating if the cancellation is performed.
  /// \param CanceledDirective The kind of directive that is cancled.
  void emitCancelationCheckImpl(Value *CancelFlag,
                                omp::Directive CanceledDirective);

  /// Generate a barrier runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  /// \param DK The directive which caused the barrier
  /// \param ForceSimpleCall Flag to force a simple (=non-cancellation) barrier.
  /// \param CheckCancelFlag Flag to indicate a cancel barrier return value
  ///                        should be checked and acted upon.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy emitBarrierImpl(const LocationDescription &Loc,
                                omp::Directive DK, bool ForceSimpleCall,
                                bool CheckCancelFlag);

  /// Generate a flush runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  void emitFlush(const LocationDescription &Loc);

  /// The finalization stack made up of finalize callbacks currently in-flight,
  /// wrapped into FinalizationInfo objects that reference also the finalization
  /// target block and the kind of cancellable directive.
  SmallVector<FinalizationInfo, 8> FinalizationStack;

  /// Return true if the last entry in the finalization stack is of kind \p DK
  /// and cancellable.
  bool isLastFinalizationInfoCancellable(omp::Directive DK) {
    return !FinalizationStack.empty() &&
           FinalizationStack.back().IsCancellable &&
           FinalizationStack.back().DK == DK;
  }

  /// Generate a taskwait runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  void emitTaskwaitImpl(const LocationDescription &Loc);

  /// Generate a taskyield runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  void emitTaskyieldImpl(const LocationDescription &Loc);

  /// Return the current thread ID.
  ///
  /// \param Ident The ident (ident_t*) describing the query origin.
  Value *getOrCreateThreadID(Value *Ident);

  /// The underlying LLVM-IR module
  Module &M;

  /// The LLVM-IR Builder used to create IR.
  IRBuilder<> Builder;

  /// Map to remember source location strings
  StringMap<Constant *> SrcLocStrMap;

  /// Map to remember existing ident_t*.
  DenseMap<std::pair<Constant *, uint64_t>, GlobalVariable *> IdentMap;

  /// Helper that contains information about regions we need to outline
  /// during finalization.
  struct OutlineInfo {
    using PostOutlineCBTy = std::function<void(Function &)>;
    PostOutlineCBTy PostOutlineCB;
    BasicBlock *EntryBB, *ExitBB;

    /// Collect all blocks in between EntryBB and ExitBB in both the given
    /// vector and set.
    void collectBlocks(SmallPtrSetImpl<BasicBlock *> &BlockSet,
                       SmallVectorImpl<BasicBlock *> &BlockVector);
  };

  /// Collection of regions that need to be outlined during finalization.
  SmallVector<OutlineInfo, 16> OutlineInfos;

  /// Add a new region that will be outlined later.
  void addOutlineInfo(OutlineInfo &&OI) { OutlineInfos.emplace_back(OI); }

  /// An ordered map of auto-generated variables to their unique names.
  /// It stores variables with the following names: 1) ".gomp_critical_user_" +
  /// <critical_section_name> + ".var" for "omp critical" directives; 2)
  /// <mangled_name_for_global_var> + ".cache." for cache for threadprivate
  /// variables.
  StringMap<AssertingVH<Constant>, BumpPtrAllocator> InternalVars;

public:
  /// Generator for '#omp master'
  ///
  /// \param Loc The insert and source location description.
  /// \param BodyGenCB Callback that will generate the region code.
  /// \param FiniCB Callback to finalize variable copies.
  ///
  /// \returns The insertion position *after* the master.
  InsertPointTy CreateMaster(const LocationDescription &Loc,
                             BodyGenCallbackTy BodyGenCB,
                             FinalizeCallbackTy FiniCB);

  /// Generator for '#omp critical'
  ///
  /// \param Loc The insert and source location description.
  /// \param BodyGenCB Callback that will generate the region body code.
  /// \param FiniCB Callback to finalize variable copies.
  /// \param CriticalName name of the lock used by the critical directive
  /// \param HintInst Hint Instruction for hint clause associated with critical
  ///
  /// \returns The insertion position *after* the master.
  InsertPointTy CreateCritical(const LocationDescription &Loc,
                               BodyGenCallbackTy BodyGenCB,
                               FinalizeCallbackTy FiniCB,
                               StringRef CriticalName, Value *HintInst);

  /// Generate conditional branch and relevant BasicBlocks through which private
  /// threads copy the 'copyin' variables from Master copy to threadprivate
  /// copies.
  ///
  /// \param IP insertion block for copyin conditional
  /// \param MasterVarPtr a pointer to the master variable
  /// \param PrivateVarPtr a pointer to the threadprivate variable
  /// \param IntPtrTy Pointer size type
  /// \param BranchtoEnd Create a branch between the copyin.not.master blocks
  //				 and copy.in.end block
  ///
  /// \returns The insertion point where copying operation to be emitted.
  InsertPointTy CreateCopyinClauseBlocks(InsertPointTy IP, Value *MasterAddr,
                                         Value *PrivateAddr,
                                         llvm::IntegerType *IntPtrTy,
                                         bool BranchtoEnd = true);

  /// Create a runtime call for kmpc_Alloc
  ///
  /// \param Loc The insert and source location description.
  /// \param Size Size of allocated memory space
  /// \param Allocator Allocator information instruction
  /// \param Name Name of call Instruction for OMP_alloc
  ///
  /// \returns CallInst to the OMP_Alloc call
  CallInst *CreateOMPAlloc(const LocationDescription &Loc, Value *Size,
                           Value *Allocator, std::string Name = "");

  /// Create a runtime call for kmpc_free
  ///
  /// \param Loc The insert and source location description.
  /// \param Addr Address of memory space to be freed
  /// \param Allocator Allocator information instruction
  /// \param Name Name of call Instruction for OMP_Free
  ///
  /// \returns CallInst to the OMP_Free call
  CallInst *CreateOMPFree(const LocationDescription &Loc, Value *Addr,
                          Value *Allocator, std::string Name = "");

  /// Create a runtime call for kmpc_threadprivate_cached
  ///
  /// \param Loc The insert and source location description.
  /// \param Pointer pointer to data to be cached
  /// \param Size size of data to be cached
  /// \param Name Name of call Instruction for callinst
  ///
  /// \returns CallInst to the thread private cache call.
  CallInst *CreateCachedThreadPrivate(const LocationDescription &Loc,
                                      llvm::Value *Pointer,
                                      llvm::ConstantInt *Size,
                                      const llvm::Twine &Name = Twine(""));

  /// Declarations for LLVM-IR types (simple, array, function and structure) are
  /// generated below. Their names are defined and used in OpenMPKinds.def. Here
  /// we provide the declarations, the initializeTypes function will provide the
  /// values.
  ///
  ///{
#define OMP_TYPE(VarName, InitValue) Type *VarName = nullptr;
#define OMP_ARRAY_TYPE(VarName, ElemTy, ArraySize)                             \
  ArrayType *VarName##Ty = nullptr;                                            \
  PointerType *VarName##PtrTy = nullptr;
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  FunctionType *VarName = nullptr;                                             \
  PointerType *VarName##Ptr = nullptr;
#define OMP_STRUCT_TYPE(VarName, StrName, ...)                                 \
  StructType *VarName = nullptr;                                               \
  PointerType *VarName##Ptr = nullptr;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  ///}

private:
  /// Create all simple and struct types exposed by the runtime and remember
  /// the llvm::PointerTypes of them for easy access later.
  void initializeTypes(Module &M);

  /// Common interface for generating entry calls for OMP Directives.
  /// if the directive has a region/body, It will set the insertion
  /// point to the body
  ///
  /// \param OMPD Directive to generate entry blocks for
  /// \param EntryCall Call to the entry OMP Runtime Function
  /// \param ExitBB block where the region ends.
  /// \param Conditional indicate if the entry call result will be used
  ///        to evaluate a conditional of whether a thread will execute
  ///        body code or not.
  ///
  /// \return The insertion position in exit block
  InsertPointTy emitCommonDirectiveEntry(omp::Directive OMPD, Value *EntryCall,
                                         BasicBlock *ExitBB,
                                         bool Conditional = false);

  /// Common interface to finalize the region
  ///
  /// \param OMPD Directive to generate exiting code for
  /// \param FinIP Insertion point for emitting Finalization code and exit call
  /// \param ExitCall Call to the ending OMP Runtime Function
  /// \param HasFinalize indicate if the directive will require finalization
  ///         and has a finalization callback in the stack that
  ///        should be called.
  ///
  /// \return The insertion position in exit block
  InsertPointTy emitCommonDirectiveExit(omp::Directive OMPD,
                                        InsertPointTy FinIP,
                                        Instruction *ExitCall,
                                        bool HasFinalize = true);

  /// Common Interface to generate OMP inlined regions
  ///
  /// \param OMPD Directive to generate inlined region for
  /// \param EntryCall Call to the entry OMP Runtime Function
  /// \param ExitCall Call to the ending OMP Runtime Function
  /// \param BodyGenCB Body code generation callback.
  /// \param FiniCB Finalization Callback. Will be called when finalizing region
  /// \param Conditional indicate if the entry call result will be used
  ///        to evaluate a conditional of whether a thread will execute
  ///        body code or not.
  /// \param HasFinalize indicate if the directive will require finalization
  ///         and has a finalization callback in the stack that
  /// should        be called.
  ///
  /// \return The insertion point after the region

  InsertPointTy
  EmitOMPInlinedRegion(omp::Directive OMPD, Instruction *EntryCall,
                       Instruction *ExitCall, BodyGenCallbackTy BodyGenCB,
                       FinalizeCallbackTy FiniCB, bool Conditional = false,
                       bool HasFinalize = true);

  /// Get the platform-specific name separator.
  /// \param Parts different parts of the final name that needs separation
  /// \param FirstSeparator First separator used between the initial two
  ///        parts of the name.
  /// \param Separator separator used between all of the rest consecutive
  ///        parts of the name
  static std::string getNameWithSeparators(ArrayRef<StringRef> Parts,
                                           StringRef FirstSeparator,
                                           StringRef Separator);

  /// Gets (if variable with the given name already exist) or creates
  /// internal global variable with the specified Name. The created variable has
  /// linkage CommonLinkage by default and is initialized by null value.
  /// \param Ty Type of the global variable. If it is exist already the type
  /// must be the same.
  /// \param Name Name of the variable.
  Constant *getOrCreateOMPInternalVariable(Type *Ty, const Twine &Name,
                                           unsigned AddressSpace = 0);

  /// Returns corresponding lock object for the specified critical region
  /// name. If the lock object does not exist it is created, otherwise the
  /// reference to the existing copy is returned.
  /// \param CriticalName Name of the critical region.
  ///
  Value *getOMPCriticalRegionLock(StringRef CriticalName);
};

} // end namespace llvm

#endif // LLVM_IR_IRBUILDER_H
