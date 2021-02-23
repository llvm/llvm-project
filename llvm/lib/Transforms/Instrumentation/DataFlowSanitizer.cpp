//===- DataFlowSanitizer.cpp - dynamic data flow analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file is a part of DataFlowSanitizer, a generalised dynamic data flow
/// analysis.
///
/// Unlike other Sanitizer tools, this tool is not designed to detect a specific
/// class of bugs on its own.  Instead, it provides a generic dynamic data flow
/// analysis framework to be used by clients to help detect application-specific
/// issues within their own code.
///
/// The analysis is based on automatic propagation of data flow labels (also
/// known as taint labels) through a program as it performs computation.  Each
/// byte of application memory is backed by two bytes of shadow memory which
/// hold the label.  On Linux/x86_64, memory is laid out as follows:
///
/// +--------------------+ 0x800000000000 (top of memory)
/// | application memory |
/// +--------------------+ 0x700000008000 (kAppAddr)
/// |                    |
/// |       unused       |
/// |                    |
/// +--------------------+ 0x300200000000 (kUnusedAddr)
/// |    union table     |
/// +--------------------+ 0x300000000000 (kUnionTableAddr)
/// |       origin       |
/// +--------------------+ 0x200000008000 (kOriginAddr)
/// |   shadow memory    |
/// +--------------------+ 0x000000010000 (kShadowAddr)
/// | reserved by kernel |
/// +--------------------+ 0x000000000000
///
/// To derive a shadow memory address from an application memory address,
/// bits 44-46 are cleared to bring the address into the range
/// [0x000000008000,0x100000000000).  Then the address is shifted left by 1 to
/// account for the double byte representation of shadow labels and move the
/// address into the shadow memory range.  See the function
/// DataFlowSanitizer::getShadowAddress below.
///
/// For more information, please refer to the design document:
/// http://clang.llvm.org/docs/DataFlowSanitizerDesign.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

// This must be consistent with ShadowWidthBits.
static const Align kShadowTLSAlignment = Align(2);

static const Align kMinOriginAlignment = Align(4);

// The size of TLS variables. These constants must be kept in sync with the ones
// in dfsan.cpp.
static const unsigned kArgTLSSize = 800;
static const unsigned kRetvalTLSSize = 800;

// External symbol to be used when generating the shadow address for
// architectures with multiple VMAs. Instead of using a constant integer
// the runtime will set the external mask based on the VMA range.
const char kDFSanExternShadowPtrMask[] = "__dfsan_shadow_ptr_mask";

// The -dfsan-preserve-alignment flag controls whether this pass assumes that
// alignment requirements provided by the input IR are correct.  For example,
// if the input IR contains a load with alignment 8, this flag will cause
// the shadow load to have alignment 16.  This flag is disabled by default as
// we have unfortunately encountered too much code (including Clang itself;
// see PR14291) which performs misaligned access.
static cl::opt<bool> ClPreserveAlignment(
    "dfsan-preserve-alignment",
    cl::desc("respect alignment requirements provided by input IR"), cl::Hidden,
    cl::init(false));

// The ABI list files control how shadow parameters are passed. The pass treats
// every function labelled "uninstrumented" in the ABI list file as conforming
// to the "native" (i.e. unsanitized) ABI.  Unless the ABI list contains
// additional annotations for those functions, a call to one of those functions
// will produce a warning message, as the labelling behaviour of the function is
// unknown.  The other supported annotations are "functional" and "discard",
// which are described below under DataFlowSanitizer::WrapperKind.
static cl::list<std::string> ClABIListFiles(
    "dfsan-abilist",
    cl::desc("File listing native ABI functions and how the pass treats them"),
    cl::Hidden);

// Controls whether the pass uses IA_Args or IA_TLS as the ABI for instrumented
// functions (see DataFlowSanitizer::InstrumentedABI below).
static cl::opt<bool> ClArgsABI(
    "dfsan-args-abi",
    cl::desc("Use the argument ABI rather than the TLS ABI"),
    cl::Hidden);

// Controls whether the pass includes or ignores the labels of pointers in load
// instructions.
static cl::opt<bool> ClCombinePointerLabelsOnLoad(
    "dfsan-combine-pointer-labels-on-load",
    cl::desc("Combine the label of the pointer with the label of the data when "
             "loading from memory."),
    cl::Hidden, cl::init(true));

// Controls whether the pass includes or ignores the labels of pointers in
// stores instructions.
static cl::opt<bool> ClCombinePointerLabelsOnStore(
    "dfsan-combine-pointer-labels-on-store",
    cl::desc("Combine the label of the pointer with the label of the data when "
             "storing in memory."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClDebugNonzeroLabels(
    "dfsan-debug-nonzero-labels",
    cl::desc("Insert calls to __dfsan_nonzero_label on observing a parameter, "
             "load or return with a nonzero label"),
    cl::Hidden);

// Experimental feature that inserts callbacks for certain data events.
// Currently callbacks are only inserted for loads, stores, memory transfers
// (i.e. memcpy and memmove), and comparisons.
//
// If this flag is set to true, the user must provide definitions for the
// following callback functions:
//   void __dfsan_load_callback(dfsan_label Label, void* addr);
//   void __dfsan_store_callback(dfsan_label Label, void* addr);
//   void __dfsan_mem_transfer_callback(dfsan_label *Start, size_t Len);
//   void __dfsan_cmp_callback(dfsan_label CombinedLabel);
static cl::opt<bool> ClEventCallbacks(
    "dfsan-event-callbacks",
    cl::desc("Insert calls to __dfsan_*_callback functions on data events."),
    cl::Hidden, cl::init(false));

// Use a distinct bit for each base label, enabling faster unions with less
// instrumentation.  Limits the max number of base labels to 16.
static cl::opt<bool> ClFast16Labels(
    "dfsan-fast-16-labels",
    cl::desc("Use more efficient instrumentation, limiting the number of "
             "labels to 16."),
    cl::Hidden, cl::init(false));

// Controls whether the pass tracks the control flow of select instructions.
static cl::opt<bool> ClTrackSelectControlFlow(
    "dfsan-track-select-control-flow",
    cl::desc("Propagate labels from condition values of select instructions "
             "to results."),
    cl::Hidden, cl::init(true));

// Controls how to track origins.
// * 0: do not track origins.
// * 1: track origins at memory store operations.
// * 2: TODO: track origins at memory store operations and callsites.
static cl::opt<int> ClTrackOrigins("dfsan-track-origins",
                                   cl::desc("Track origins of labels"),
                                   cl::Hidden, cl::init(0));

static StringRef GetGlobalTypeString(const GlobalValue &G) {
  // Types of GlobalVariables are always pointer types.
  Type *GType = G.getValueType();
  // For now we support excluding struct types only.
  if (StructType *SGType = dyn_cast<StructType>(GType)) {
    if (!SGType->isLiteral())
      return SGType->getName();
  }
  return "<unknown type>";
}

namespace {

class DFSanABIList {
  std::unique_ptr<SpecialCaseList> SCL;

 public:
  DFSanABIList() = default;

  void set(std::unique_ptr<SpecialCaseList> List) { SCL = std::move(List); }

  /// Returns whether either this function or its source file are listed in the
  /// given category.
  bool isIn(const Function &F, StringRef Category) const {
    return isIn(*F.getParent(), Category) ||
           SCL->inSection("dataflow", "fun", F.getName(), Category);
  }

  /// Returns whether this global alias is listed in the given category.
  ///
  /// If GA aliases a function, the alias's name is matched as a function name
  /// would be.  Similarly, aliases of globals are matched like globals.
  bool isIn(const GlobalAlias &GA, StringRef Category) const {
    if (isIn(*GA.getParent(), Category))
      return true;

    if (isa<FunctionType>(GA.getValueType()))
      return SCL->inSection("dataflow", "fun", GA.getName(), Category);

    return SCL->inSection("dataflow", "global", GA.getName(), Category) ||
           SCL->inSection("dataflow", "type", GetGlobalTypeString(GA),
                          Category);
  }

  /// Returns whether this module is listed in the given category.
  bool isIn(const Module &M, StringRef Category) const {
    return SCL->inSection("dataflow", "src", M.getModuleIdentifier(), Category);
  }
};

/// TransformedFunction is used to express the result of transforming one
/// function type into another.  This struct is immutable.  It holds metadata
/// useful for updating calls of the old function to the new type.
struct TransformedFunction {
  TransformedFunction(FunctionType* OriginalType,
                      FunctionType* TransformedType,
                      std::vector<unsigned> ArgumentIndexMapping)
      : OriginalType(OriginalType),
        TransformedType(TransformedType),
        ArgumentIndexMapping(ArgumentIndexMapping) {}

  // Disallow copies.
  TransformedFunction(const TransformedFunction&) = delete;
  TransformedFunction& operator=(const TransformedFunction&) = delete;

  // Allow moves.
  TransformedFunction(TransformedFunction&&) = default;
  TransformedFunction& operator=(TransformedFunction&&) = default;

  /// Type of the function before the transformation.
  FunctionType *OriginalType;

  /// Type of the function after the transformation.
  FunctionType *TransformedType;

  /// Transforming a function may change the position of arguments.  This
  /// member records the mapping from each argument's old position to its new
  /// position.  Argument positions are zero-indexed.  If the transformation
  /// from F to F' made the first argument of F into the third argument of F',
  /// then ArgumentIndexMapping[0] will equal 2.
  std::vector<unsigned> ArgumentIndexMapping;
};

/// Given function attributes from a call site for the original function,
/// return function attributes appropriate for a call to the transformed
/// function.
AttributeList TransformFunctionAttributes(
    const TransformedFunction& TransformedFunction,
    LLVMContext& Ctx, AttributeList CallSiteAttrs) {

  // Construct a vector of AttributeSet for each function argument.
  std::vector<llvm::AttributeSet> ArgumentAttributes(
      TransformedFunction.TransformedType->getNumParams());

  // Copy attributes from the parameter of the original function to the
  // transformed version.  'ArgumentIndexMapping' holds the mapping from
  // old argument position to new.
  for (unsigned i=0, ie = TransformedFunction.ArgumentIndexMapping.size();
       i < ie; ++i) {
    unsigned TransformedIndex = TransformedFunction.ArgumentIndexMapping[i];
    ArgumentAttributes[TransformedIndex] = CallSiteAttrs.getParamAttributes(i);
  }

  // Copy annotations on varargs arguments.
  for (unsigned i = TransformedFunction.OriginalType->getNumParams(),
       ie = CallSiteAttrs.getNumAttrSets(); i<ie; ++i) {
    ArgumentAttributes.push_back(CallSiteAttrs.getParamAttributes(i));
  }

  return AttributeList::get(
      Ctx,
      CallSiteAttrs.getFnAttributes(),
      CallSiteAttrs.getRetAttributes(),
      llvm::makeArrayRef(ArgumentAttributes));
}

class DataFlowSanitizer {
  friend struct DFSanFunction;
  friend class DFSanVisitor;

  enum {
    ShadowWidthBits = 16,
    ShadowWidthBytes = ShadowWidthBits / 8,
    OriginWidthBits = 32,
    OriginWidthBytes = OriginWidthBits / 8
  };

  /// Which ABI should be used for instrumented functions?
  enum InstrumentedABI {
    /// Argument and return value labels are passed through additional
    /// arguments and by modifying the return type.
    IA_Args,

    /// Argument and return value labels are passed through TLS variables
    /// __dfsan_arg_tls and __dfsan_retval_tls.
    IA_TLS
  };

  /// How should calls to uninstrumented functions be handled?
  enum WrapperKind {
    /// This function is present in an uninstrumented form but we don't know
    /// how it should be handled.  Print a warning and call the function anyway.
    /// Don't label the return value.
    WK_Warning,

    /// This function does not write to (user-accessible) memory, and its return
    /// value is unlabelled.
    WK_Discard,

    /// This function does not write to (user-accessible) memory, and the label
    /// of its return value is the union of the label of its arguments.
    WK_Functional,

    /// Instead of calling the function, a custom wrapper __dfsw_F is called,
    /// where F is the name of the function.  This function may wrap the
    /// original function or provide its own implementation.  This is similar to
    /// the IA_Args ABI, except that IA_Args uses a struct return type to
    /// pass the return value shadow in a register, while WK_Custom uses an
    /// extra pointer argument to return the shadow.  This allows the wrapped
    /// form of the function type to be expressed in C.
    WK_Custom
  };

  Module *Mod;
  LLVMContext *Ctx;
  Type *Int8Ptr;
  IntegerType *OriginTy;
  PointerType *OriginPtrTy;
  ConstantInt *OriginBase;
  ConstantInt *ZeroOrigin;
  /// The shadow type for all primitive types and vector types.
  IntegerType *PrimitiveShadowTy;
  PointerType *PrimitiveShadowPtrTy;
  IntegerType *IntptrTy;
  ConstantInt *ZeroPrimitiveShadow;
  ConstantInt *ShadowPtrMask;
  ConstantInt *ShadowPtrMul;
  Constant *ArgTLS;
  ArrayType *ArgOriginTLSTy;
  Constant *ArgOriginTLS;
  Constant *RetvalTLS;
  Constant *RetvalOriginTLS;
  Constant *ExternalShadowMask;
  FunctionType *DFSanUnionFnTy;
  FunctionType *DFSanUnionLoadFnTy;
  FunctionType *DFSanLoadLabelAndOriginFnTy;
  FunctionType *DFSanUnimplementedFnTy;
  FunctionType *DFSanSetLabelFnTy;
  FunctionType *DFSanNonzeroLabelFnTy;
  FunctionType *DFSanVarargWrapperFnTy;
  FunctionType *DFSanCmpCallbackFnTy;
  FunctionType *DFSanLoadStoreCallbackFnTy;
  FunctionType *DFSanMemTransferCallbackFnTy;
  FunctionType *DFSanChainOriginFnTy;
  FunctionType *DFSanMemOriginTransferFnTy;
  FunctionType *DFSanMaybeStoreOriginFnTy;
  FunctionCallee DFSanUnionFn;
  FunctionCallee DFSanCheckedUnionFn;
  FunctionCallee DFSanUnionLoadFn;
  FunctionCallee DFSanUnionLoadFast16LabelsFn;
  FunctionCallee DFSanLoadLabelAndOriginFn;
  FunctionCallee DFSanUnimplementedFn;
  FunctionCallee DFSanSetLabelFn;
  FunctionCallee DFSanNonzeroLabelFn;
  FunctionCallee DFSanVarargWrapperFn;
  FunctionCallee DFSanLoadCallbackFn;
  FunctionCallee DFSanStoreCallbackFn;
  FunctionCallee DFSanMemTransferCallbackFn;
  FunctionCallee DFSanCmpCallbackFn;
  FunctionCallee DFSanChainOriginFn;
  FunctionCallee DFSanMemOriginTransferFn;
  FunctionCallee DFSanMaybeStoreOriginFn;
  SmallPtrSet<Value *, 16> DFSanRuntimeFunctions;
  MDNode *ColdCallWeights;
  DFSanABIList ABIList;
  DenseMap<Value *, Function *> UnwrappedFnMap;
  AttrBuilder ReadOnlyNoneAttrs;
  bool DFSanRuntimeShadowMask = false;

  Value *getShadowOffset(Value *Addr, IRBuilder<> &IRB);
  Value *getShadowAddress(Value *Addr, Instruction *Pos);
  // std::pair<Value *, Value *>
  // getShadowOriginAddress(Value *Addr, Align InstAlignment, Instruction *Pos);
  bool isInstrumented(const Function *F);
  bool isInstrumented(const GlobalAlias *GA);
  FunctionType *getArgsFunctionType(FunctionType *T);
  FunctionType *getTrampolineFunctionType(FunctionType *T);
  TransformedFunction getCustomFunctionType(FunctionType *T);
  InstrumentedABI getInstrumentedABI();
  WrapperKind getWrapperKind(Function *F);
  void addGlobalNamePrefix(GlobalValue *GV);
  Function *buildWrapperFunction(Function *F, StringRef NewFName,
                                 GlobalValue::LinkageTypes NewFLink,
                                 FunctionType *NewFT);
  Constant *getOrBuildTrampolineFunction(FunctionType *FT, StringRef FName);
  void initializeCallbackFunctions(Module &M);
  void initializeRuntimeFunctions(Module &M);

  bool init(Module &M);

  /// Returns whether the pass tracks origins. Support only fast16 mode in TLS
  /// ABI mode.
  bool shouldTrackOrigins();

  /// Returns whether the pass tracks labels for struct fields and array
  /// indices. Support only fast16 mode in TLS ABI mode.
  bool shouldTrackFieldsAndIndices();

  /// Returns a zero constant with the shadow type of OrigTy.
  ///
  /// getZeroShadow({T1,T2,...}) = {getZeroShadow(T1),getZeroShadow(T2,...}
  /// getZeroShadow([n x T]) = [n x getZeroShadow(T)]
  /// getZeroShadow(other type) = i16(0)
  ///
  /// Note that a zero shadow is always i16(0) when shouldTrackFieldsAndIndices
  /// returns false.
  Constant *getZeroShadow(Type *OrigTy);
  /// Returns a zero constant with the shadow type of V's type.
  Constant *getZeroShadow(Value *V);

  /// Checks if V is a zero shadow.
  bool isZeroShadow(Value *V);

  /// Returns the shadow type of OrigTy.
  ///
  /// getShadowTy({T1,T2,...}) = {getShadowTy(T1),getShadowTy(T2),...}
  /// getShadowTy([n x T]) = [n x getShadowTy(T)]
  /// getShadowTy(other type) = i16
  ///
  /// Note that a shadow type is always i16 when shouldTrackFieldsAndIndices
  /// returns false.
  Type *getShadowTy(Type *OrigTy);
  /// Returns the shadow type of of V's type.
  Type *getShadowTy(Value *V);

  const uint64_t kNumOfElementsInArgOrgTLS = kArgTLSSize / OriginWidthBytes;

public:
  DataFlowSanitizer(const std::vector<std::string> &ABIListFiles);

  bool runImpl(Module &M);
};

struct DFSanFunction {
  DataFlowSanitizer &DFS;
  Function *F;
  DominatorTree DT;
  DataFlowSanitizer::InstrumentedABI IA;
  bool IsNativeABI;
  AllocaInst *LabelReturnAlloca = nullptr;
  DenseMap<Value *, Value *> ValShadowMap;
  DenseMap<Value *, Value *> ValOriginMap;
  DenseMap<AllocaInst *, AllocaInst *> AllocaShadowMap;
  std::vector<std::pair<PHINode *, PHINode *>> PHIFixups;
  DenseSet<Instruction *> SkipInsts;
  std::vector<Value *> NonZeroChecks;
  bool AvoidNewBlocks;

  struct CachedShadow {
    BasicBlock *Block; // The block where Shadow is defined.
    Value *Shadow;
  };
  /// Maps a value to its latest shadow value in terms of domination tree.
  DenseMap<std::pair<Value *, Value *>, CachedShadow> CachedShadows;
  /// Maps a value to its latest collapsed shadow value it was converted to in
  /// terms of domination tree. When ClDebugNonzeroLabels is on, this cache is
  /// used at a post process where CFG blocks are split. So it does not cache
  /// BasicBlock like CachedShadows, but uses domination between values.
  DenseMap<Value *, Value *> CachedCollapsedShadows;
  DenseMap<Value *, std::set<Value *>> ShadowElements;

  DFSanFunction(DataFlowSanitizer &DFS, Function *F, bool IsNativeABI)
      : DFS(DFS), F(F), IA(DFS.getInstrumentedABI()), IsNativeABI(IsNativeABI) {
    DT.recalculate(*F);
    // FIXME: Need to track down the register allocator issue which causes poor
    // performance in pathological cases with large numbers of basic blocks.
    AvoidNewBlocks = F->size() > 1000;
  }

  /// Computes the shadow address for a given function argument.
  ///
  /// Shadow = ArgTLS+ArgOffset.
  Value *getArgTLS(Type *T, unsigned ArgOffset, IRBuilder<> &IRB);

  /// Computes the shadow address for a return value.
  Value *getRetvalTLS(Type *T, IRBuilder<> &IRB);

  /// Computes the origin address for a given function argument.
  ///
  /// Origin = ArgOriginTLS[ArgNo].
  Value *getArgOriginTLS(unsigned ArgNo, IRBuilder<> &IRB);

  /// Computes the origin address for a return value.
  Value *getRetvalOriginTLS();

  Value *getOrigin(Value *V);
  void setOrigin(Instruction *I, Value *Origin);
  /// Generates IR to compute the origin of the last operand with a taint label.
  Value *combineOperandOrigins(Instruction *Inst);
  /// Before the instruction Pos, generates IR to compute the last origin with a
  /// taint label. Labels and origins are from vectors Shadows and Origins
  /// correspondingly. The generated IR is like
  ///   Sn-1 != Zero ? On-1: ... S2 != Zero ? O2: S1 != Zero ? O1: O0
  /// When Zero is nullptr, it uses ZeroPrimitiveShadow. Otherwise it can be
  /// zeros with other bitwidths.
  Value *combineOrigins(const std::vector<Value *> &Shadows,
                        const std::vector<Value *> &Origins, Instruction *Pos,
                        ConstantInt *Zero = nullptr);

  Value *getShadow(Value *V);
  void setShadow(Instruction *I, Value *Shadow);
  /// Generates IR to compute the union of the two given shadows, inserting it
  /// before Pos. The combined value is with primitive type.
  Value *combineShadows(Value *V1, Value *V2, Instruction *Pos);
  /// Combines the shadow values of V1 and V2, then converts the combined value
  /// with primitive type into a shadow value with the original type T.
  Value *combineShadowsThenConvert(Type *T, Value *V1, Value *V2,
                                   Instruction *Pos);
  Value *combineOperandShadows(Instruction *Inst);
  Value *loadShadow(Value *ShadowAddr, uint64_t Size, uint64_t Align,
                    Instruction *Pos);
  void storePrimitiveShadow(Value *Addr, uint64_t Size, Align Alignment,
                            Value *PrimitiveShadow, Instruction *Pos);
  /// Applies PrimitiveShadow to all primitive subtypes of T, returning
  /// the expanded shadow value.
  ///
  /// EFP({T1,T2, ...}, PS) = {EFP(T1,PS),EFP(T2,PS),...}
  /// EFP([n x T], PS) = [n x EFP(T,PS)]
  /// EFP(other types, PS) = PS
  Value *expandFromPrimitiveShadow(Type *T, Value *PrimitiveShadow,
                                   Instruction *Pos);
  /// Collapses Shadow into a single primitive shadow value, unioning all
  /// primitive shadow values in the process. Returns the final primitive
  /// shadow value.
  ///
  /// CTP({V1,V2, ...}) = UNION(CFP(V1,PS),CFP(V2,PS),...)
  /// CTP([V1,V2,...]) = UNION(CFP(V1,PS),CFP(V2,PS),...)
  /// CTP(other types, PS) = PS
  Value *collapseToPrimitiveShadow(Value *Shadow, Instruction *Pos);

private:
  /// Collapses the shadow with aggregate type into a single primitive shadow
  /// value.
  template <class AggregateType>
  Value *collapseAggregateShadow(AggregateType *AT, Value *Shadow,
                                 IRBuilder<> &IRB);

  Value *collapseToPrimitiveShadow(Value *Shadow, IRBuilder<> &IRB);

  /// Returns the shadow value of an argument A.
  Value *getShadowForTLSArgument(Argument *A);

  /// The fast path of loading shadow in legacy mode.
  Value *loadLegacyShadowFast(Value *ShadowAddr, uint64_t Size,
                              Align ShadowAlign, Instruction *Pos);

  /// The fast path of loading shadow in fast-16-label mode.
  Value *loadFast16ShadowFast(Value *ShadowAddr, uint64_t Size,
                              Align ShadowAlign, Instruction *Pos);
};

class DFSanVisitor : public InstVisitor<DFSanVisitor> {
public:
  DFSanFunction &DFSF;

  DFSanVisitor(DFSanFunction &DFSF) : DFSF(DFSF) {}

  const DataLayout &getDataLayout() const {
    return DFSF.F->getParent()->getDataLayout();
  }

  // Combines shadow values and origins for all of I's operands.
  void visitInstOperands(Instruction &I);

  void visitUnaryOperator(UnaryOperator &UO);
  void visitBinaryOperator(BinaryOperator &BO);
  void visitCastInst(CastInst &CI);
  void visitCmpInst(CmpInst &CI);
  void visitGetElementPtrInst(GetElementPtrInst &GEPI);
  void visitLoadInst(LoadInst &LI);
  void visitStoreInst(StoreInst &SI);
  void visitReturnInst(ReturnInst &RI);
  void visitCallBase(CallBase &CB);
  void visitPHINode(PHINode &PN);
  void visitExtractElementInst(ExtractElementInst &I);
  void visitInsertElementInst(InsertElementInst &I);
  void visitShuffleVectorInst(ShuffleVectorInst &I);
  void visitExtractValueInst(ExtractValueInst &I);
  void visitInsertValueInst(InsertValueInst &I);
  void visitAllocaInst(AllocaInst &I);
  void visitSelectInst(SelectInst &I);
  void visitMemSetInst(MemSetInst &I);
  void visitMemTransferInst(MemTransferInst &I);

private:
  // Returns false when this is an invoke of a custom function.
  bool visitWrappedCallBase(Function &F, CallBase &CB);

  // Combines origins for all of I's operands.
  void visitInstOperandOrigins(Instruction &I);
};

} // end anonymous namespace

DataFlowSanitizer::DataFlowSanitizer(
    const std::vector<std::string> &ABIListFiles) {
  std::vector<std::string> AllABIListFiles(std::move(ABIListFiles));
  llvm::append_range(AllABIListFiles, ClABIListFiles);
  // FIXME: should we propagate vfs::FileSystem to this constructor?
  ABIList.set(
      SpecialCaseList::createOrDie(AllABIListFiles, *vfs::getRealFileSystem()));
}

FunctionType *DataFlowSanitizer::getArgsFunctionType(FunctionType *T) {
  SmallVector<Type *, 4> ArgTypes(T->param_begin(), T->param_end());
  ArgTypes.append(T->getNumParams(), PrimitiveShadowTy);
  if (T->isVarArg())
    ArgTypes.push_back(PrimitiveShadowPtrTy);
  Type *RetType = T->getReturnType();
  if (!RetType->isVoidTy())
    RetType = StructType::get(RetType, PrimitiveShadowTy);
  return FunctionType::get(RetType, ArgTypes, T->isVarArg());
}

FunctionType *DataFlowSanitizer::getTrampolineFunctionType(FunctionType *T) {
  assert(!T->isVarArg());
  SmallVector<Type *, 4> ArgTypes;
  ArgTypes.push_back(T->getPointerTo());
  ArgTypes.append(T->param_begin(), T->param_end());
  ArgTypes.append(T->getNumParams(), PrimitiveShadowTy);
  Type *RetType = T->getReturnType();
  if (!RetType->isVoidTy())
    ArgTypes.push_back(PrimitiveShadowPtrTy);
  return FunctionType::get(T->getReturnType(), ArgTypes, false);
}

TransformedFunction DataFlowSanitizer::getCustomFunctionType(FunctionType *T) {
  SmallVector<Type *, 4> ArgTypes;

  // Some parameters of the custom function being constructed are
  // parameters of T.  Record the mapping from parameters of T to
  // parameters of the custom function, so that parameter attributes
  // at call sites can be updated.
  std::vector<unsigned> ArgumentIndexMapping;
  for (unsigned i = 0, ie = T->getNumParams(); i != ie; ++i) {
    Type* param_type = T->getParamType(i);
    FunctionType *FT;
    if (isa<PointerType>(param_type) && (FT = dyn_cast<FunctionType>(
            cast<PointerType>(param_type)->getElementType()))) {
      ArgumentIndexMapping.push_back(ArgTypes.size());
      ArgTypes.push_back(getTrampolineFunctionType(FT)->getPointerTo());
      ArgTypes.push_back(Type::getInt8PtrTy(*Ctx));
    } else {
      ArgumentIndexMapping.push_back(ArgTypes.size());
      ArgTypes.push_back(param_type);
    }
  }
  for (unsigned i = 0, e = T->getNumParams(); i != e; ++i)
    ArgTypes.push_back(PrimitiveShadowTy);
  if (T->isVarArg())
    ArgTypes.push_back(PrimitiveShadowPtrTy);
  Type *RetType = T->getReturnType();
  if (!RetType->isVoidTy())
    ArgTypes.push_back(PrimitiveShadowPtrTy);
  return TransformedFunction(
      T, FunctionType::get(T->getReturnType(), ArgTypes, T->isVarArg()),
      ArgumentIndexMapping);
}

bool DataFlowSanitizer::isZeroShadow(Value *V) {
  if (!shouldTrackFieldsAndIndices())
    return ZeroPrimitiveShadow == V;

  Type *T = V->getType();
  if (!isa<ArrayType>(T) && !isa<StructType>(T)) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return CI->isZero();
    return false;
  }

  return isa<ConstantAggregateZero>(V);
}

bool DataFlowSanitizer::shouldTrackOrigins() {
  return ClTrackOrigins && getInstrumentedABI() == DataFlowSanitizer::IA_TLS &&
         ClFast16Labels;
}

bool DataFlowSanitizer::shouldTrackFieldsAndIndices() {
  return getInstrumentedABI() == DataFlowSanitizer::IA_TLS && ClFast16Labels;
}

Constant *DataFlowSanitizer::getZeroShadow(Type *OrigTy) {
  if (!shouldTrackFieldsAndIndices())
    return ZeroPrimitiveShadow;

  if (!isa<ArrayType>(OrigTy) && !isa<StructType>(OrigTy))
    return ZeroPrimitiveShadow;
  Type *ShadowTy = getShadowTy(OrigTy);
  return ConstantAggregateZero::get(ShadowTy);
}

Constant *DataFlowSanitizer::getZeroShadow(Value *V) {
  return getZeroShadow(V->getType());
}

static Value *expandFromPrimitiveShadowRecursive(
    Value *Shadow, SmallVector<unsigned, 4> &Indices, Type *SubShadowTy,
    Value *PrimitiveShadow, IRBuilder<> &IRB) {
  if (!isa<ArrayType>(SubShadowTy) && !isa<StructType>(SubShadowTy))
    return IRB.CreateInsertValue(Shadow, PrimitiveShadow, Indices);

  if (ArrayType *AT = dyn_cast<ArrayType>(SubShadowTy)) {
    for (unsigned Idx = 0; Idx < AT->getNumElements(); Idx++) {
      Indices.push_back(Idx);
      Shadow = expandFromPrimitiveShadowRecursive(
          Shadow, Indices, AT->getElementType(), PrimitiveShadow, IRB);
      Indices.pop_back();
    }
    return Shadow;
  }

  if (StructType *ST = dyn_cast<StructType>(SubShadowTy)) {
    for (unsigned Idx = 0; Idx < ST->getNumElements(); Idx++) {
      Indices.push_back(Idx);
      Shadow = expandFromPrimitiveShadowRecursive(
          Shadow, Indices, ST->getElementType(Idx), PrimitiveShadow, IRB);
      Indices.pop_back();
    }
    return Shadow;
  }
  llvm_unreachable("Unexpected shadow type");
}

Value *DFSanFunction::expandFromPrimitiveShadow(Type *T, Value *PrimitiveShadow,
                                                Instruction *Pos) {
  Type *ShadowTy = DFS.getShadowTy(T);

  if (!isa<ArrayType>(ShadowTy) && !isa<StructType>(ShadowTy))
    return PrimitiveShadow;

  if (DFS.isZeroShadow(PrimitiveShadow))
    return DFS.getZeroShadow(ShadowTy);

  IRBuilder<> IRB(Pos);
  SmallVector<unsigned, 4> Indices;
  Value *Shadow = UndefValue::get(ShadowTy);
  Shadow = expandFromPrimitiveShadowRecursive(Shadow, Indices, ShadowTy,
                                              PrimitiveShadow, IRB);

  // Caches the primitive shadow value that built the shadow value.
  CachedCollapsedShadows[Shadow] = PrimitiveShadow;
  return Shadow;
}

template <class AggregateType>
Value *DFSanFunction::collapseAggregateShadow(AggregateType *AT, Value *Shadow,
                                              IRBuilder<> &IRB) {
  if (!AT->getNumElements())
    return DFS.ZeroPrimitiveShadow;

  Value *FirstItem = IRB.CreateExtractValue(Shadow, 0);
  Value *Aggregator = collapseToPrimitiveShadow(FirstItem, IRB);

  for (unsigned Idx = 1; Idx < AT->getNumElements(); Idx++) {
    Value *ShadowItem = IRB.CreateExtractValue(Shadow, Idx);
    Value *ShadowInner = collapseToPrimitiveShadow(ShadowItem, IRB);
    Aggregator = IRB.CreateOr(Aggregator, ShadowInner);
  }
  return Aggregator;
}

Value *DFSanFunction::collapseToPrimitiveShadow(Value *Shadow,
                                                IRBuilder<> &IRB) {
  Type *ShadowTy = Shadow->getType();
  if (!isa<ArrayType>(ShadowTy) && !isa<StructType>(ShadowTy))
    return Shadow;
  if (ArrayType *AT = dyn_cast<ArrayType>(ShadowTy))
    return collapseAggregateShadow<>(AT, Shadow, IRB);
  if (StructType *ST = dyn_cast<StructType>(ShadowTy))
    return collapseAggregateShadow<>(ST, Shadow, IRB);
  llvm_unreachable("Unexpected shadow type");
}

Value *DFSanFunction::collapseToPrimitiveShadow(Value *Shadow,
                                                Instruction *Pos) {
  Type *ShadowTy = Shadow->getType();
  if (!isa<ArrayType>(ShadowTy) && !isa<StructType>(ShadowTy))
    return Shadow;

  assert(DFS.shouldTrackFieldsAndIndices());

  // Checks if the cached collapsed shadow value dominates Pos.
  Value *&CS = CachedCollapsedShadows[Shadow];
  if (CS && DT.dominates(CS, Pos))
    return CS;

  IRBuilder<> IRB(Pos);
  Value *PrimitiveShadow = collapseToPrimitiveShadow(Shadow, IRB);
  // Caches the converted primitive shadow value.
  CS = PrimitiveShadow;
  return PrimitiveShadow;
}

Type *DataFlowSanitizer::getShadowTy(Type *OrigTy) {
  if (!shouldTrackFieldsAndIndices())
    return PrimitiveShadowTy;

  if (!OrigTy->isSized())
    return PrimitiveShadowTy;
  if (isa<IntegerType>(OrigTy))
    return PrimitiveShadowTy;
  if (isa<VectorType>(OrigTy))
    return PrimitiveShadowTy;
  if (ArrayType *AT = dyn_cast<ArrayType>(OrigTy))
    return ArrayType::get(getShadowTy(AT->getElementType()),
                          AT->getNumElements());
  if (StructType *ST = dyn_cast<StructType>(OrigTy)) {
    SmallVector<Type *, 4> Elements;
    for (unsigned I = 0, N = ST->getNumElements(); I < N; ++I)
      Elements.push_back(getShadowTy(ST->getElementType(I)));
    return StructType::get(*Ctx, Elements);
  }
  return PrimitiveShadowTy;
}

Type *DataFlowSanitizer::getShadowTy(Value *V) {
  return getShadowTy(V->getType());
}

bool DataFlowSanitizer::init(Module &M) {
  Triple TargetTriple(M.getTargetTriple());
  bool IsX86_64 = TargetTriple.getArch() == Triple::x86_64;
  bool IsMIPS64 = TargetTriple.isMIPS64();
  bool IsAArch64 = TargetTriple.getArch() == Triple::aarch64 ||
                   TargetTriple.getArch() == Triple::aarch64_be;

  const DataLayout &DL = M.getDataLayout();

  Mod = &M;
  Ctx = &M.getContext();
  Int8Ptr = Type::getInt8PtrTy(*Ctx);
  OriginTy = IntegerType::get(*Ctx, OriginWidthBits);
  OriginPtrTy = PointerType::getUnqual(OriginTy);
  PrimitiveShadowTy = IntegerType::get(*Ctx, ShadowWidthBits);
  PrimitiveShadowPtrTy = PointerType::getUnqual(PrimitiveShadowTy);
  IntptrTy = DL.getIntPtrType(*Ctx);
  ZeroPrimitiveShadow = ConstantInt::getSigned(PrimitiveShadowTy, 0);
  ShadowPtrMul = ConstantInt::getSigned(IntptrTy, ShadowWidthBytes);
  OriginBase = ConstantInt::get(IntptrTy, 0x200000000000LL);
  ZeroOrigin = ConstantInt::getSigned(OriginTy, 0);
  if (IsX86_64)
    ShadowPtrMask = ConstantInt::getSigned(IntptrTy, ~0x700000000000LL);
  else if (IsMIPS64)
    ShadowPtrMask = ConstantInt::getSigned(IntptrTy, ~0xF000000000LL);
  // AArch64 supports multiple VMAs and the shadow mask is set at runtime.
  else if (IsAArch64)
    DFSanRuntimeShadowMask = true;
  else
    report_fatal_error("unsupported triple");

  Type *DFSanUnionArgs[2] = {PrimitiveShadowTy, PrimitiveShadowTy};
  DFSanUnionFnTy =
      FunctionType::get(PrimitiveShadowTy, DFSanUnionArgs, /*isVarArg=*/false);
  Type *DFSanUnionLoadArgs[2] = {PrimitiveShadowPtrTy, IntptrTy};
  DFSanUnionLoadFnTy = FunctionType::get(PrimitiveShadowTy, DFSanUnionLoadArgs,
                                         /*isVarArg=*/false);
  Type *DFSanLoadLabelAndOriginArgs[2] = {Int8Ptr, IntptrTy};
  DFSanLoadLabelAndOriginFnTy =
      FunctionType::get(IntegerType::get(*Ctx, 64), DFSanLoadLabelAndOriginArgs,
                        /*isVarArg=*/false);
  DFSanUnimplementedFnTy = FunctionType::get(
      Type::getVoidTy(*Ctx), Type::getInt8PtrTy(*Ctx), /*isVarArg=*/false);
  Type *DFSanSetLabelArgs[3] = {PrimitiveShadowTy, Type::getInt8PtrTy(*Ctx),
                                IntptrTy};
  DFSanSetLabelFnTy = FunctionType::get(Type::getVoidTy(*Ctx),
                                        DFSanSetLabelArgs, /*isVarArg=*/false);
  DFSanNonzeroLabelFnTy =
      FunctionType::get(Type::getVoidTy(*Ctx), None, /*isVarArg=*/false);
  DFSanVarargWrapperFnTy = FunctionType::get(
      Type::getVoidTy(*Ctx), Type::getInt8PtrTy(*Ctx), /*isVarArg=*/false);
  DFSanCmpCallbackFnTy =
      FunctionType::get(Type::getVoidTy(*Ctx), PrimitiveShadowTy,
                        /*isVarArg=*/false);
  DFSanChainOriginFnTy =
      FunctionType::get(OriginTy, OriginTy, /*isVarArg=*/false);
  Type *DFSanMaybeStoreOriginArgs[4] = {IntegerType::get(*Ctx, ShadowWidthBits),
                                        Int8Ptr, IntptrTy, OriginTy};
  DFSanMaybeStoreOriginFnTy = FunctionType::get(
      Type::getVoidTy(*Ctx), DFSanMaybeStoreOriginArgs, /*isVarArg=*/false);
  Type *DFSanMemOriginTransferArgs[3] = {Int8Ptr, Int8Ptr, IntptrTy};
  DFSanMemOriginTransferFnTy = FunctionType::get(
      Type::getVoidTy(*Ctx), DFSanMemOriginTransferArgs, /*isVarArg=*/false);
  Type *DFSanLoadStoreCallbackArgs[2] = {PrimitiveShadowTy, Int8Ptr};
  DFSanLoadStoreCallbackFnTy =
      FunctionType::get(Type::getVoidTy(*Ctx), DFSanLoadStoreCallbackArgs,
                        /*isVarArg=*/false);
  Type *DFSanMemTransferCallbackArgs[2] = {PrimitiveShadowPtrTy, IntptrTy};
  DFSanMemTransferCallbackFnTy =
      FunctionType::get(Type::getVoidTy(*Ctx), DFSanMemTransferCallbackArgs,
                        /*isVarArg=*/false);

  ColdCallWeights = MDBuilder(*Ctx).createBranchWeights(1, 1000);
  return true;
}

bool DataFlowSanitizer::isInstrumented(const Function *F) {
  return !ABIList.isIn(*F, "uninstrumented");
}

bool DataFlowSanitizer::isInstrumented(const GlobalAlias *GA) {
  return !ABIList.isIn(*GA, "uninstrumented");
}

DataFlowSanitizer::InstrumentedABI DataFlowSanitizer::getInstrumentedABI() {
  return ClArgsABI ? IA_Args : IA_TLS;
}

DataFlowSanitizer::WrapperKind DataFlowSanitizer::getWrapperKind(Function *F) {
  if (ABIList.isIn(*F, "functional"))
    return WK_Functional;
  if (ABIList.isIn(*F, "discard"))
    return WK_Discard;
  if (ABIList.isIn(*F, "custom"))
    return WK_Custom;

  return WK_Warning;
}

void DataFlowSanitizer::addGlobalNamePrefix(GlobalValue *GV) {
  std::string GVName = std::string(GV->getName()), Prefix = "dfs$";
  GV->setName(Prefix + GVName);

  // Try to change the name of the function in module inline asm.  We only do
  // this for specific asm directives, currently only ".symver", to try to avoid
  // corrupting asm which happens to contain the symbol name as a substring.
  // Note that the substitution for .symver assumes that the versioned symbol
  // also has an instrumented name.
  std::string Asm = GV->getParent()->getModuleInlineAsm();
  std::string SearchStr = ".symver " + GVName + ",";
  size_t Pos = Asm.find(SearchStr);
  if (Pos != std::string::npos) {
    Asm.replace(Pos, SearchStr.size(),
                ".symver " + Prefix + GVName + "," + Prefix);
    GV->getParent()->setModuleInlineAsm(Asm);
  }
}

Function *
DataFlowSanitizer::buildWrapperFunction(Function *F, StringRef NewFName,
                                        GlobalValue::LinkageTypes NewFLink,
                                        FunctionType *NewFT) {
  FunctionType *FT = F->getFunctionType();
  Function *NewF = Function::Create(NewFT, NewFLink, F->getAddressSpace(),
                                    NewFName, F->getParent());
  NewF->copyAttributesFrom(F);
  NewF->removeAttributes(
      AttributeList::ReturnIndex,
      AttributeFuncs::typeIncompatible(NewFT->getReturnType()));

  BasicBlock *BB = BasicBlock::Create(*Ctx, "entry", NewF);
  if (F->isVarArg()) {
    NewF->removeAttributes(AttributeList::FunctionIndex,
                           AttrBuilder().addAttribute("split-stack"));
    CallInst::Create(DFSanVarargWrapperFn,
                     IRBuilder<>(BB).CreateGlobalStringPtr(F->getName()), "",
                     BB);
    new UnreachableInst(*Ctx, BB);
  } else {
    std::vector<Value *> Args;
    unsigned n = FT->getNumParams();
    for (Function::arg_iterator ai = NewF->arg_begin(); n != 0; ++ai, --n)
      Args.push_back(&*ai);
    CallInst *CI = CallInst::Create(F, Args, "", BB);
    if (FT->getReturnType()->isVoidTy())
      ReturnInst::Create(*Ctx, BB);
    else
      ReturnInst::Create(*Ctx, CI, BB);
  }

  return NewF;
}

Constant *DataFlowSanitizer::getOrBuildTrampolineFunction(FunctionType *FT,
                                                          StringRef FName) {
  FunctionType *FTT = getTrampolineFunctionType(FT);
  FunctionCallee C = Mod->getOrInsertFunction(FName, FTT);
  Function *F = dyn_cast<Function>(C.getCallee());
  if (F && F->isDeclaration()) {
    F->setLinkage(GlobalValue::LinkOnceODRLinkage);
    BasicBlock *BB = BasicBlock::Create(*Ctx, "entry", F);
    std::vector<Value *> Args;
    Function::arg_iterator AI = F->arg_begin(); ++AI;
    for (unsigned N = FT->getNumParams(); N != 0; ++AI, --N)
      Args.push_back(&*AI);
    CallInst *CI = CallInst::Create(FT, &*F->arg_begin(), Args, "", BB);
    ReturnInst *RI;
    if (FT->getReturnType()->isVoidTy())
      RI = ReturnInst::Create(*Ctx, BB);
    else
      RI = ReturnInst::Create(*Ctx, CI, BB);

    // F is called by a wrapped custom function with primitive shadows. So
    // its arguments and return value need conversion.
    DFSanFunction DFSF(*this, F, /*IsNativeABI=*/true);
    Function::arg_iterator ValAI = F->arg_begin(), ShadowAI = AI; ++ValAI;
    for (unsigned N = FT->getNumParams(); N != 0; ++ValAI, ++ShadowAI, --N) {
      Value *Shadow =
          DFSF.expandFromPrimitiveShadow(ValAI->getType(), &*ShadowAI, CI);
      DFSF.ValShadowMap[&*ValAI] = Shadow;
    }
    DFSanVisitor(DFSF).visitCallInst(*CI);
    if (!FT->getReturnType()->isVoidTy()) {
      Value *PrimitiveShadow = DFSF.collapseToPrimitiveShadow(
          DFSF.getShadow(RI->getReturnValue()), RI);
      new StoreInst(PrimitiveShadow, &*std::prev(F->arg_end()), RI);
    }
  }

  return cast<Constant>(C.getCallee());
}

// Initialize DataFlowSanitizer runtime functions and declare them in the module
void DataFlowSanitizer::initializeRuntimeFunctions(Module &M) {
  {
    AttributeList AL;
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::NoUnwind);
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::ReadNone);
    AL = AL.addAttribute(M.getContext(), AttributeList::ReturnIndex,
                         Attribute::ZExt);
    AL = AL.addParamAttribute(M.getContext(), 0, Attribute::ZExt);
    AL = AL.addParamAttribute(M.getContext(), 1, Attribute::ZExt);
    DFSanUnionFn =
        Mod->getOrInsertFunction("__dfsan_union", DFSanUnionFnTy, AL);
  }
  {
    AttributeList AL;
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::NoUnwind);
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::ReadNone);
    AL = AL.addAttribute(M.getContext(), AttributeList::ReturnIndex,
                         Attribute::ZExt);
    AL = AL.addParamAttribute(M.getContext(), 0, Attribute::ZExt);
    AL = AL.addParamAttribute(M.getContext(), 1, Attribute::ZExt);
    DFSanCheckedUnionFn =
        Mod->getOrInsertFunction("dfsan_union", DFSanUnionFnTy, AL);
  }
  {
    AttributeList AL;
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::NoUnwind);
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::ReadOnly);
    AL = AL.addAttribute(M.getContext(), AttributeList::ReturnIndex,
                         Attribute::ZExt);
    DFSanUnionLoadFn =
        Mod->getOrInsertFunction("__dfsan_union_load", DFSanUnionLoadFnTy, AL);
  }
  {
    AttributeList AL;
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::NoUnwind);
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::ReadOnly);
    AL = AL.addAttribute(M.getContext(), AttributeList::ReturnIndex,
                         Attribute::ZExt);
    DFSanUnionLoadFast16LabelsFn = Mod->getOrInsertFunction(
        "__dfsan_union_load_fast16labels", DFSanUnionLoadFnTy, AL);
  }
  {
    AttributeList AL;
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::NoUnwind);
    AL = AL.addAttribute(M.getContext(), AttributeList::FunctionIndex,
                         Attribute::ReadOnly);
    AL = AL.addAttribute(M.getContext(), AttributeList::ReturnIndex,
                         Attribute::ZExt);
    DFSanLoadLabelAndOriginFn = Mod->getOrInsertFunction(
        "__dfsan_load_label_and_origin", DFSanLoadLabelAndOriginFnTy, AL);
  }
  DFSanUnimplementedFn =
      Mod->getOrInsertFunction("__dfsan_unimplemented", DFSanUnimplementedFnTy);
  {
    AttributeList AL;
    AL = AL.addParamAttribute(M.getContext(), 0, Attribute::ZExt);
    DFSanSetLabelFn =
        Mod->getOrInsertFunction("__dfsan_set_label", DFSanSetLabelFnTy, AL);
  }
  DFSanNonzeroLabelFn =
      Mod->getOrInsertFunction("__dfsan_nonzero_label", DFSanNonzeroLabelFnTy);
  DFSanVarargWrapperFn = Mod->getOrInsertFunction("__dfsan_vararg_wrapper",
                                                  DFSanVarargWrapperFnTy);
  {
    AttributeList AL;
    AL = AL.addParamAttribute(M.getContext(), 0, Attribute::ZExt);
    AL = AL.addAttribute(M.getContext(), AttributeList::ReturnIndex,
                         Attribute::ZExt);
    DFSanChainOriginFn = Mod->getOrInsertFunction("__dfsan_chain_origin",
                                                  DFSanChainOriginFnTy, AL);
  }
  DFSanMemOriginTransferFn = Mod->getOrInsertFunction(
      "__dfsan_mem_origin_transfer", DFSanMemOriginTransferFnTy);

  {
    AttributeList AL;
    AL = AL.addParamAttribute(M.getContext(), 0, Attribute::ZExt);
    AL = AL.addParamAttribute(M.getContext(), 3, Attribute::ZExt);
    DFSanMaybeStoreOriginFn = Mod->getOrInsertFunction(
        "__dfsan_maybe_store_origin", DFSanMaybeStoreOriginFnTy, AL);
  }

  DFSanRuntimeFunctions.insert(DFSanUnionFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanCheckedUnionFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanUnionLoadFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanUnionLoadFast16LabelsFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanLoadLabelAndOriginFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanUnimplementedFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanSetLabelFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanNonzeroLabelFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanVarargWrapperFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanLoadCallbackFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanStoreCallbackFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanMemTransferCallbackFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanCmpCallbackFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanChainOriginFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanMemOriginTransferFn.getCallee()->stripPointerCasts());
  DFSanRuntimeFunctions.insert(
      DFSanMaybeStoreOriginFn.getCallee()->stripPointerCasts());
}

// Initializes event callback functions and declare them in the module
void DataFlowSanitizer::initializeCallbackFunctions(Module &M) {
  DFSanLoadCallbackFn = Mod->getOrInsertFunction("__dfsan_load_callback",
                                                 DFSanLoadStoreCallbackFnTy);
  DFSanStoreCallbackFn = Mod->getOrInsertFunction("__dfsan_store_callback",
                                                  DFSanLoadStoreCallbackFnTy);
  DFSanMemTransferCallbackFn = Mod->getOrInsertFunction(
      "__dfsan_mem_transfer_callback", DFSanMemTransferCallbackFnTy);
  DFSanCmpCallbackFn =
      Mod->getOrInsertFunction("__dfsan_cmp_callback", DFSanCmpCallbackFnTy);
}

bool DataFlowSanitizer::runImpl(Module &M) {
  init(M);

  if (ABIList.isIn(M, "skip"))
    return false;

  const unsigned InitialGlobalSize = M.global_size();
  const unsigned InitialModuleSize = M.size();

  bool Changed = false;

  auto getOrInsertGlobal = [this, &Changed](StringRef Name,
                                            Type *Ty) -> Constant * {
    Constant *C = Mod->getOrInsertGlobal(Name, Ty);
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(C)) {
      Changed |= G->getThreadLocalMode() != GlobalVariable::InitialExecTLSModel;
      G->setThreadLocalMode(GlobalVariable::InitialExecTLSModel);
    }
    return C;
  };

  // These globals must be kept in sync with the ones in dfsan.cpp.
  ArgTLS = getOrInsertGlobal(
      "__dfsan_arg_tls",
      ArrayType::get(Type::getInt64Ty(*Ctx), kArgTLSSize / 8));
  RetvalTLS = getOrInsertGlobal(
      "__dfsan_retval_tls",
      ArrayType::get(Type::getInt64Ty(*Ctx), kRetvalTLSSize / 8));
  ArgOriginTLSTy = ArrayType::get(OriginTy, kNumOfElementsInArgOrgTLS);
  ArgOriginTLS = getOrInsertGlobal("__dfsan_arg_origin_tls", ArgOriginTLSTy);
  RetvalOriginTLS = getOrInsertGlobal("__dfsan_retval_origin_tls", OriginTy);

  (void)Mod->getOrInsertGlobal("__dfsan_track_origins", OriginTy, [&] {
    Changed = true;
    return new GlobalVariable(
        M, OriginTy, true, GlobalValue::WeakODRLinkage,
        ConstantInt::getSigned(OriginTy, shouldTrackOrigins()),
        "__dfsan_track_origins");
  });

  ExternalShadowMask =
      Mod->getOrInsertGlobal(kDFSanExternShadowPtrMask, IntptrTy);

  initializeCallbackFunctions(M);
  initializeRuntimeFunctions(M);

  std::vector<Function *> FnsToInstrument;
  SmallPtrSet<Function *, 2> FnsWithNativeABI;
  for (Function &i : M)
    if (!i.isIntrinsic() && !DFSanRuntimeFunctions.contains(&i))
      FnsToInstrument.push_back(&i);

  // Give function aliases prefixes when necessary, and build wrappers where the
  // instrumentedness is inconsistent.
  for (Module::alias_iterator i = M.alias_begin(), e = M.alias_end(); i != e;) {
    GlobalAlias *GA = &*i;
    ++i;
    // Don't stop on weak.  We assume people aren't playing games with the
    // instrumentedness of overridden weak aliases.
    if (auto F = dyn_cast<Function>(GA->getBaseObject())) {
      bool GAInst = isInstrumented(GA), FInst = isInstrumented(F);
      if (GAInst && FInst) {
        addGlobalNamePrefix(GA);
      } else if (GAInst != FInst) {
        // Non-instrumented alias of an instrumented function, or vice versa.
        // Replace the alias with a native-ABI wrapper of the aliasee.  The pass
        // below will take care of instrumenting it.
        Function *NewF =
            buildWrapperFunction(F, "", GA->getLinkage(), F->getFunctionType());
        GA->replaceAllUsesWith(ConstantExpr::getBitCast(NewF, GA->getType()));
        NewF->takeName(GA);
        GA->eraseFromParent();
        FnsToInstrument.push_back(NewF);
      }
    }
  }

  ReadOnlyNoneAttrs.addAttribute(Attribute::ReadOnly)
      .addAttribute(Attribute::ReadNone);

  // First, change the ABI of every function in the module.  ABI-listed
  // functions keep their original ABI and get a wrapper function.
  for (std::vector<Function *>::iterator i = FnsToInstrument.begin(),
                                         e = FnsToInstrument.end();
       i != e; ++i) {
    Function &F = **i;
    FunctionType *FT = F.getFunctionType();

    bool IsZeroArgsVoidRet = (FT->getNumParams() == 0 && !FT->isVarArg() &&
                              FT->getReturnType()->isVoidTy());

    if (isInstrumented(&F)) {
      // Instrumented functions get a 'dfs$' prefix.  This allows us to more
      // easily identify cases of mismatching ABIs.
      if (getInstrumentedABI() == IA_Args && !IsZeroArgsVoidRet) {
        FunctionType *NewFT = getArgsFunctionType(FT);
        Function *NewF = Function::Create(NewFT, F.getLinkage(),
                                          F.getAddressSpace(), "", &M);
        NewF->copyAttributesFrom(&F);
        NewF->removeAttributes(
            AttributeList::ReturnIndex,
            AttributeFuncs::typeIncompatible(NewFT->getReturnType()));
        for (Function::arg_iterator FArg = F.arg_begin(),
                                    NewFArg = NewF->arg_begin(),
                                    FArgEnd = F.arg_end();
             FArg != FArgEnd; ++FArg, ++NewFArg) {
          FArg->replaceAllUsesWith(&*NewFArg);
        }
        NewF->getBasicBlockList().splice(NewF->begin(), F.getBasicBlockList());

        for (Function::user_iterator UI = F.user_begin(), UE = F.user_end();
             UI != UE;) {
          BlockAddress *BA = dyn_cast<BlockAddress>(*UI);
          ++UI;
          if (BA) {
            BA->replaceAllUsesWith(
                BlockAddress::get(NewF, BA->getBasicBlock()));
            delete BA;
          }
        }
        F.replaceAllUsesWith(
            ConstantExpr::getBitCast(NewF, PointerType::getUnqual(FT)));
        NewF->takeName(&F);
        F.eraseFromParent();
        *i = NewF;
        addGlobalNamePrefix(NewF);
      } else {
        addGlobalNamePrefix(&F);
      }
    } else if (!IsZeroArgsVoidRet || getWrapperKind(&F) == WK_Custom) {
      // Build a wrapper function for F.  The wrapper simply calls F, and is
      // added to FnsToInstrument so that any instrumentation according to its
      // WrapperKind is done in the second pass below.
      FunctionType *NewFT = getInstrumentedABI() == IA_Args
                                ? getArgsFunctionType(FT)
                                : FT;

      // If the function being wrapped has local linkage, then preserve the
      // function's linkage in the wrapper function.
      GlobalValue::LinkageTypes wrapperLinkage =
          F.hasLocalLinkage()
              ? F.getLinkage()
              : GlobalValue::LinkOnceODRLinkage;

      Function *NewF = buildWrapperFunction(
          &F, std::string("dfsw$") + std::string(F.getName()),
          wrapperLinkage, NewFT);
      if (getInstrumentedABI() == IA_TLS)
        NewF->removeAttributes(AttributeList::FunctionIndex, ReadOnlyNoneAttrs);

      Value *WrappedFnCst =
          ConstantExpr::getBitCast(NewF, PointerType::getUnqual(FT));
      F.replaceAllUsesWith(WrappedFnCst);

      UnwrappedFnMap[WrappedFnCst] = &F;
      *i = NewF;

      if (!F.isDeclaration()) {
        // This function is probably defining an interposition of an
        // uninstrumented function and hence needs to keep the original ABI.
        // But any functions it may call need to use the instrumented ABI, so
        // we instrument it in a mode which preserves the original ABI.
        FnsWithNativeABI.insert(&F);

        // This code needs to rebuild the iterators, as they may be invalidated
        // by the push_back, taking care that the new range does not include
        // any functions added by this code.
        size_t N = i - FnsToInstrument.begin(),
               Count = e - FnsToInstrument.begin();
        FnsToInstrument.push_back(&F);
        i = FnsToInstrument.begin() + N;
        e = FnsToInstrument.begin() + Count;
      }
               // Hopefully, nobody will try to indirectly call a vararg
               // function... yet.
    } else if (FT->isVarArg()) {
      UnwrappedFnMap[&F] = &F;
      *i = nullptr;
    }
  }

  for (Function *i : FnsToInstrument) {
    if (!i || i->isDeclaration())
      continue;

    removeUnreachableBlocks(*i);

    DFSanFunction DFSF(*this, i, FnsWithNativeABI.count(i));

    // DFSanVisitor may create new basic blocks, which confuses df_iterator.
    // Build a copy of the list before iterating over it.
    SmallVector<BasicBlock *, 4> BBList(depth_first(&i->getEntryBlock()));

    for (BasicBlock *i : BBList) {
      Instruction *Inst = &i->front();
      while (true) {
        // DFSanVisitor may split the current basic block, changing the current
        // instruction's next pointer and moving the next instruction to the
        // tail block from which we should continue.
        Instruction *Next = Inst->getNextNode();
        // DFSanVisitor may delete Inst, so keep track of whether it was a
        // terminator.
        bool IsTerminator = Inst->isTerminator();
        if (!DFSF.SkipInsts.count(Inst))
          DFSanVisitor(DFSF).visit(Inst);
        if (IsTerminator)
          break;
        Inst = Next;
      }
    }

    // We will not necessarily be able to compute the shadow for every phi node
    // until we have visited every block.  Therefore, the code that handles phi
    // nodes adds them to the PHIFixups list so that they can be properly
    // handled here.
    for (std::vector<std::pair<PHINode *, PHINode *>>::iterator
             i = DFSF.PHIFixups.begin(),
             e = DFSF.PHIFixups.end();
         i != e; ++i) {
      for (unsigned val = 0, n = i->first->getNumIncomingValues(); val != n;
           ++val) {
        i->second->setIncomingValue(
            val, DFSF.getShadow(i->first->getIncomingValue(val)));
      }
    }

    // -dfsan-debug-nonzero-labels will split the CFG in all kinds of crazy
    // places (i.e. instructions in basic blocks we haven't even begun visiting
    // yet).  To make our life easier, do this work in a pass after the main
    // instrumentation.
    if (ClDebugNonzeroLabels) {
      for (Value *V : DFSF.NonZeroChecks) {
        Instruction *Pos;
        if (Instruction *I = dyn_cast<Instruction>(V))
          Pos = I->getNextNode();
        else
          Pos = &DFSF.F->getEntryBlock().front();
        while (isa<PHINode>(Pos) || isa<AllocaInst>(Pos))
          Pos = Pos->getNextNode();
        IRBuilder<> IRB(Pos);
        Value *PrimitiveShadow = DFSF.collapseToPrimitiveShadow(V, Pos);
        Value *Ne =
            IRB.CreateICmpNE(PrimitiveShadow, DFSF.DFS.ZeroPrimitiveShadow);
        BranchInst *BI = cast<BranchInst>(SplitBlockAndInsertIfThen(
            Ne, Pos, /*Unreachable=*/false, ColdCallWeights));
        IRBuilder<> ThenIRB(BI);
        ThenIRB.CreateCall(DFSF.DFS.DFSanNonzeroLabelFn, {});
      }
    }
  }

  return Changed || !FnsToInstrument.empty() ||
         M.global_size() != InitialGlobalSize || M.size() != InitialModuleSize;
}

Value *DFSanFunction::getArgTLS(Type *T, unsigned ArgOffset, IRBuilder<> &IRB) {
  Value *Base = IRB.CreatePointerCast(DFS.ArgTLS, DFS.IntptrTy);
  if (ArgOffset)
    Base = IRB.CreateAdd(Base, ConstantInt::get(DFS.IntptrTy, ArgOffset));
  return IRB.CreateIntToPtr(Base, PointerType::get(DFS.getShadowTy(T), 0),
                            "_dfsarg");
}

Value *DFSanFunction::getRetvalTLS(Type *T, IRBuilder<> &IRB) {
  return IRB.CreatePointerCast(
      DFS.RetvalTLS, PointerType::get(DFS.getShadowTy(T), 0), "_dfsret");
}

Value *DFSanFunction::getRetvalOriginTLS() { return DFS.RetvalOriginTLS; }

Value *DFSanFunction::getArgOriginTLS(unsigned ArgNo, IRBuilder<> &IRB) {
  return IRB.CreateConstGEP2_64(DFS.ArgOriginTLSTy, DFS.ArgOriginTLS, 0, ArgNo,
                                "_dfsarg_o");
}

Value *DFSanFunction::getOrigin(Value *V) {
  assert(DFS.shouldTrackOrigins());
  if (!isa<Argument>(V) && !isa<Instruction>(V))
    return DFS.ZeroOrigin;
  Value *&Origin = ValOriginMap[V];
  if (!Origin) {
    if (Argument *A = dyn_cast<Argument>(V)) {
      if (IsNativeABI)
        return DFS.ZeroOrigin;
      switch (IA) {
      case DataFlowSanitizer::IA_TLS: {
        if (A->getArgNo() < DFS.kNumOfElementsInArgOrgTLS) {
          Instruction *ArgOriginTLSPos = &*F->getEntryBlock().begin();
          IRBuilder<> IRB(ArgOriginTLSPos);
          Value *ArgOriginPtr = getArgOriginTLS(A->getArgNo(), IRB);
          Origin = IRB.CreateLoad(DFS.OriginTy, ArgOriginPtr);
        } else {
          // Overflow
          Origin = DFS.ZeroOrigin;
        }
        break;
      }
      case DataFlowSanitizer::IA_Args: {
        Origin = DFS.ZeroOrigin;
        break;
      }
      }
    } else {
      Origin = DFS.ZeroOrigin;
    }
  }
  return Origin;
}

void DFSanFunction::setOrigin(Instruction *I, Value *Origin) {
  if (!DFS.shouldTrackOrigins())
    return;
  assert(!ValOriginMap.count(I));
  assert(Origin->getType() == DFS.OriginTy);
  ValOriginMap[I] = Origin;
}

Value *DFSanFunction::getShadowForTLSArgument(Argument *A) {
  unsigned ArgOffset = 0;
  const DataLayout &DL = F->getParent()->getDataLayout();
  for (auto &FArg : F->args()) {
    if (!FArg.getType()->isSized()) {
      if (A == &FArg)
        break;
      continue;
    }

    unsigned Size = DL.getTypeAllocSize(DFS.getShadowTy(&FArg));
    if (A != &FArg) {
      ArgOffset += alignTo(Size, kShadowTLSAlignment);
      if (ArgOffset > kArgTLSSize)
        break; // ArgTLS overflows, uses a zero shadow.
      continue;
    }

    if (ArgOffset + Size > kArgTLSSize)
      break; // ArgTLS overflows, uses a zero shadow.

    Instruction *ArgTLSPos = &*F->getEntryBlock().begin();
    IRBuilder<> IRB(ArgTLSPos);
    Value *ArgShadowPtr = getArgTLS(FArg.getType(), ArgOffset, IRB);
    return IRB.CreateAlignedLoad(DFS.getShadowTy(&FArg), ArgShadowPtr,
                                 kShadowTLSAlignment);
  }

  return DFS.getZeroShadow(A);
}

Value *DFSanFunction::getShadow(Value *V) {
  if (!isa<Argument>(V) && !isa<Instruction>(V))
    return DFS.getZeroShadow(V);
  Value *&Shadow = ValShadowMap[V];
  if (!Shadow) {
    if (Argument *A = dyn_cast<Argument>(V)) {
      if (IsNativeABI)
        return DFS.getZeroShadow(V);
      switch (IA) {
      case DataFlowSanitizer::IA_TLS: {
        Shadow = getShadowForTLSArgument(A);
        break;
      }
      case DataFlowSanitizer::IA_Args: {
        unsigned ArgIdx = A->getArgNo() + F->arg_size() / 2;
        Function::arg_iterator i = F->arg_begin();
        while (ArgIdx--)
          ++i;
        Shadow = &*i;
        assert(Shadow->getType() == DFS.PrimitiveShadowTy);
        break;
      }
      }
      NonZeroChecks.push_back(Shadow);
    } else {
      Shadow = DFS.getZeroShadow(V);
    }
  }
  return Shadow;
}

void DFSanFunction::setShadow(Instruction *I, Value *Shadow) {
  assert(!ValShadowMap.count(I));
  assert(DFS.shouldTrackFieldsAndIndices() ||
         Shadow->getType() == DFS.PrimitiveShadowTy);
  ValShadowMap[I] = Shadow;
}

Value *DataFlowSanitizer::getShadowOffset(Value *Addr, IRBuilder<> &IRB) {
  // Returns Addr & shadow_mask
  assert(Addr != RetvalTLS && "Reinstrumenting?");
  Value *ShadowPtrMaskValue;
  if (DFSanRuntimeShadowMask)
    ShadowPtrMaskValue = IRB.CreateLoad(IntptrTy, ExternalShadowMask);
  else
    ShadowPtrMaskValue = ShadowPtrMask;
  return IRB.CreateAnd(IRB.CreatePtrToInt(Addr, IntptrTy),
                       IRB.CreatePtrToInt(ShadowPtrMaskValue, IntptrTy));
}
/*
std::pair<Value *, Value *>
DataFlowSanitizer::getShadowOriginAddress(Value *Addr, Align InstAlignment,
                                          Instruction *Pos) {
  // Returns ((Addr & shadow_mask) + origin_base) & ~4UL
  IRBuilder<> IRB(Pos);
  Value *ShadowOffset = getShadowOffset(Addr, IRB);
  Value *ShadowPtr = IRB.CreateIntToPtr(
      IRB.CreateMul(ShadowOffset, ShadowPtrMul), PrimitiveShadowPtrTy);
  Value *OriginPtr = nullptr;
  if (shouldTrackOrigins()) {
    Value *OriginLong = IRB.CreateAdd(ShadowOffset, OriginBase);
    const Align Alignment = llvm::assumeAligned(InstAlignment.value());
    // When alignment is >= 4, Addr must be aligned to 4, otherwise it is UB.
    // So Mask is unnecessary.
    if (Alignment < kMinOriginAlignment) {
      uint64_t Mask = kMinOriginAlignment.value() - 1;
      OriginLong = IRB.CreateAnd(OriginLong, ConstantInt::get(IntptrTy, ~Mask));
    }
    OriginPtr = IRB.CreateIntToPtr(OriginLong, OriginPtrTy);
  }
  return {ShadowPtr, OriginPtr};
}
*/
Value *DataFlowSanitizer::getShadowAddress(Value *Addr, Instruction *Pos) {
  // Returns (Addr & shadow_mask) x 2
  IRBuilder<> IRB(Pos);
  Value *ShadowOffset = getShadowOffset(Addr, IRB);
  return IRB.CreateIntToPtr(IRB.CreateMul(ShadowOffset, ShadowPtrMul),
                            PrimitiveShadowPtrTy);
}

Value *DFSanFunction::combineShadowsThenConvert(Type *T, Value *V1, Value *V2,
                                                Instruction *Pos) {
  Value *PrimitiveValue = combineShadows(V1, V2, Pos);
  return expandFromPrimitiveShadow(T, PrimitiveValue, Pos);
}

// Generates IR to compute the union of the two given shadows, inserting it
// before Pos. The combined value is with primitive type.
Value *DFSanFunction::combineShadows(Value *V1, Value *V2, Instruction *Pos) {
  if (DFS.isZeroShadow(V1))
    return collapseToPrimitiveShadow(V2, Pos);
  if (DFS.isZeroShadow(V2))
    return collapseToPrimitiveShadow(V1, Pos);
  if (V1 == V2)
    return collapseToPrimitiveShadow(V1, Pos);

  auto V1Elems = ShadowElements.find(V1);
  auto V2Elems = ShadowElements.find(V2);
  if (V1Elems != ShadowElements.end() && V2Elems != ShadowElements.end()) {
    if (std::includes(V1Elems->second.begin(), V1Elems->second.end(),
                      V2Elems->second.begin(), V2Elems->second.end())) {
      return collapseToPrimitiveShadow(V1, Pos);
    } else if (std::includes(V2Elems->second.begin(), V2Elems->second.end(),
                             V1Elems->second.begin(), V1Elems->second.end())) {
      return collapseToPrimitiveShadow(V2, Pos);
    }
  } else if (V1Elems != ShadowElements.end()) {
    if (V1Elems->second.count(V2))
      return collapseToPrimitiveShadow(V1, Pos);
  } else if (V2Elems != ShadowElements.end()) {
    if (V2Elems->second.count(V1))
      return collapseToPrimitiveShadow(V2, Pos);
  }

  auto Key = std::make_pair(V1, V2);
  if (V1 > V2)
    std::swap(Key.first, Key.second);
  CachedShadow &CCS = CachedShadows[Key];
  if (CCS.Block && DT.dominates(CCS.Block, Pos->getParent()))
    return CCS.Shadow;

  // Converts inputs shadows to shadows with primitive types.
  Value *PV1 = collapseToPrimitiveShadow(V1, Pos);
  Value *PV2 = collapseToPrimitiveShadow(V2, Pos);

  IRBuilder<> IRB(Pos);
  if (ClFast16Labels) {
    CCS.Block = Pos->getParent();
    CCS.Shadow = IRB.CreateOr(PV1, PV2);
  } else if (AvoidNewBlocks) {
    CallInst *Call = IRB.CreateCall(DFS.DFSanCheckedUnionFn, {PV1, PV2});
    Call->addAttribute(AttributeList::ReturnIndex, Attribute::ZExt);
    Call->addParamAttr(0, Attribute::ZExt);
    Call->addParamAttr(1, Attribute::ZExt);

    CCS.Block = Pos->getParent();
    CCS.Shadow = Call;
  } else {
    BasicBlock *Head = Pos->getParent();
    Value *Ne = IRB.CreateICmpNE(PV1, PV2);
    BranchInst *BI = cast<BranchInst>(SplitBlockAndInsertIfThen(
        Ne, Pos, /*Unreachable=*/false, DFS.ColdCallWeights, &DT));
    IRBuilder<> ThenIRB(BI);
    CallInst *Call = ThenIRB.CreateCall(DFS.DFSanUnionFn, {PV1, PV2});
    Call->addAttribute(AttributeList::ReturnIndex, Attribute::ZExt);
    Call->addParamAttr(0, Attribute::ZExt);
    Call->addParamAttr(1, Attribute::ZExt);

    BasicBlock *Tail = BI->getSuccessor(0);
    PHINode *Phi =
        PHINode::Create(DFS.PrimitiveShadowTy, 2, "", &Tail->front());
    Phi->addIncoming(Call, Call->getParent());
    Phi->addIncoming(PV1, Head);

    CCS.Block = Tail;
    CCS.Shadow = Phi;
  }

  std::set<Value *> UnionElems;
  if (V1Elems != ShadowElements.end()) {
    UnionElems = V1Elems->second;
  } else {
    UnionElems.insert(V1);
  }
  if (V2Elems != ShadowElements.end()) {
    UnionElems.insert(V2Elems->second.begin(), V2Elems->second.end());
  } else {
    UnionElems.insert(V2);
  }
  ShadowElements[CCS.Shadow] = std::move(UnionElems);

  return CCS.Shadow;
}

// A convenience function which folds the shadows of each of the operands
// of the provided instruction Inst, inserting the IR before Inst.  Returns
// the computed union Value.
Value *DFSanFunction::combineOperandShadows(Instruction *Inst) {
  if (Inst->getNumOperands() == 0)
    return DFS.getZeroShadow(Inst);

  Value *Shadow = getShadow(Inst->getOperand(0));
  for (unsigned i = 1, n = Inst->getNumOperands(); i != n; ++i) {
    Shadow = combineShadows(Shadow, getShadow(Inst->getOperand(i)), Inst);
  }
  return expandFromPrimitiveShadow(Inst->getType(), Shadow, Inst);
}

void DFSanVisitor::visitInstOperands(Instruction &I) {
  Value *CombinedShadow = DFSF.combineOperandShadows(&I);
  DFSF.setShadow(&I, CombinedShadow);
  visitInstOperandOrigins(I);
}

Value *DFSanFunction::combineOrigins(const std::vector<Value *> &Shadows,
                                     const std::vector<Value *> &Origins,
                                     Instruction *Pos, ConstantInt *Zero) {
  assert(Shadows.size() == Origins.size());
  size_t Size = Origins.size();
  if (Size == 0)
    return DFS.ZeroOrigin;
  Value *Origin = nullptr;
  if (!Zero)
    Zero = DFS.ZeroPrimitiveShadow;
  for (size_t I = 0; I != Size; ++I) {
    Value *OpOrigin = Origins[I];
    Constant *ConstOpOrigin = dyn_cast<Constant>(OpOrigin);
    if (ConstOpOrigin && ConstOpOrigin->isNullValue())
      continue;
    if (!Origin) {
      Origin = OpOrigin;
      continue;
    }
    Value *OpShadow = Shadows[I];
    Value *PrimitiveShadow = collapseToPrimitiveShadow(OpShadow, Pos);
    IRBuilder<> IRB(Pos);
    Value *Cond = IRB.CreateICmpNE(PrimitiveShadow, Zero);
    Origin = IRB.CreateSelect(Cond, OpOrigin, Origin);
  }
  return Origin ? Origin : DFS.ZeroOrigin;
}

Value *DFSanFunction::combineOperandOrigins(Instruction *Inst) {
  size_t Size = Inst->getNumOperands();
  std::vector<Value *> Shadows(Size);
  std::vector<Value *> Origins(Size);
  for (unsigned I = 0; I != Size; ++I) {
    Shadows[I] = getShadow(Inst->getOperand(I));
    Origins[I] = getOrigin(Inst->getOperand(I));
  }
  return combineOrigins(Shadows, Origins, Inst);
}

void DFSanVisitor::visitInstOperandOrigins(Instruction &I) {
  if (!DFSF.DFS.shouldTrackOrigins())
    return;
  Value *CombinedOrigin = DFSF.combineOperandOrigins(&I);
  DFSF.setOrigin(&I, CombinedOrigin);
}

Value *DFSanFunction::loadFast16ShadowFast(Value *ShadowAddr, uint64_t Size,
                                           Align ShadowAlign,
                                           Instruction *Pos) {
  // First OR all the WideShadows, then OR individual shadows within the
  // combined WideShadow. This is fewer instructions than ORing shadows
  // individually.
  IRBuilder<> IRB(Pos);
  Value *WideAddr =
      IRB.CreateBitCast(ShadowAddr, Type::getInt64PtrTy(*DFS.Ctx));
  Value *CombinedWideShadow =
      IRB.CreateAlignedLoad(IRB.getInt64Ty(), WideAddr, ShadowAlign);
  for (uint64_t Ofs = 64 / DFS.ShadowWidthBits; Ofs != Size;
       Ofs += 64 / DFS.ShadowWidthBits) {
    WideAddr = IRB.CreateGEP(Type::getInt64Ty(*DFS.Ctx), WideAddr,
                             ConstantInt::get(DFS.IntptrTy, 1));
    Value *NextWideShadow =
        IRB.CreateAlignedLoad(IRB.getInt64Ty(), WideAddr, ShadowAlign);
    CombinedWideShadow = IRB.CreateOr(CombinedWideShadow, NextWideShadow);
  }
  for (unsigned Width = 32; Width >= DFS.ShadowWidthBits; Width >>= 1) {
    Value *ShrShadow = IRB.CreateLShr(CombinedWideShadow, Width);
    CombinedWideShadow = IRB.CreateOr(CombinedWideShadow, ShrShadow);
  }
  return IRB.CreateTrunc(CombinedWideShadow, DFS.PrimitiveShadowTy);
}

Value *DFSanFunction::loadLegacyShadowFast(Value *ShadowAddr, uint64_t Size,
                                           Align ShadowAlign,
                                           Instruction *Pos) {
  // Fast path for the common case where each byte has identical shadow: load
  // shadow 64 bits at a time, fall out to a __dfsan_union_load call if any
  // shadow is non-equal.
  BasicBlock *FallbackBB = BasicBlock::Create(*DFS.Ctx, "", F);
  IRBuilder<> FallbackIRB(FallbackBB);
  CallInst *FallbackCall = FallbackIRB.CreateCall(
      DFS.DFSanUnionLoadFn, {ShadowAddr, ConstantInt::get(DFS.IntptrTy, Size)});
  FallbackCall->addAttribute(AttributeList::ReturnIndex, Attribute::ZExt);

  // Compare each of the shadows stored in the loaded 64 bits to each other,
  // by computing (WideShadow rotl ShadowWidthBits) == WideShadow.
  IRBuilder<> IRB(Pos);
  Value *WideAddr =
      IRB.CreateBitCast(ShadowAddr, Type::getInt64PtrTy(*DFS.Ctx));
  Value *WideShadow =
      IRB.CreateAlignedLoad(IRB.getInt64Ty(), WideAddr, ShadowAlign);
  Value *TruncShadow = IRB.CreateTrunc(WideShadow, DFS.PrimitiveShadowTy);
  Value *ShlShadow = IRB.CreateShl(WideShadow, DFS.ShadowWidthBits);
  Value *ShrShadow = IRB.CreateLShr(WideShadow, 64 - DFS.ShadowWidthBits);
  Value *RotShadow = IRB.CreateOr(ShlShadow, ShrShadow);
  Value *ShadowsEq = IRB.CreateICmpEQ(WideShadow, RotShadow);

  BasicBlock *Head = Pos->getParent();
  BasicBlock *Tail = Head->splitBasicBlock(Pos->getIterator());

  if (DomTreeNode *OldNode = DT.getNode(Head)) {
    std::vector<DomTreeNode *> Children(OldNode->begin(), OldNode->end());

    DomTreeNode *NewNode = DT.addNewBlock(Tail, Head);
    for (auto *Child : Children)
      DT.changeImmediateDominator(Child, NewNode);
  }

  // In the following code LastBr will refer to the previous basic block's
  // conditional branch instruction, whose true successor is fixed up to point
  // to the next block during the loop below or to the tail after the final
  // iteration.
  BranchInst *LastBr = BranchInst::Create(FallbackBB, FallbackBB, ShadowsEq);
  ReplaceInstWithInst(Head->getTerminator(), LastBr);
  DT.addNewBlock(FallbackBB, Head);

  for (uint64_t Ofs = 64 / DFS.ShadowWidthBits; Ofs != Size;
       Ofs += 64 / DFS.ShadowWidthBits) {
    BasicBlock *NextBB = BasicBlock::Create(*DFS.Ctx, "", F);
    DT.addNewBlock(NextBB, LastBr->getParent());
    IRBuilder<> NextIRB(NextBB);
    WideAddr = NextIRB.CreateGEP(Type::getInt64Ty(*DFS.Ctx), WideAddr,
                                 ConstantInt::get(DFS.IntptrTy, 1));
    Value *NextWideShadow =
        NextIRB.CreateAlignedLoad(NextIRB.getInt64Ty(), WideAddr, ShadowAlign);
    ShadowsEq = NextIRB.CreateICmpEQ(WideShadow, NextWideShadow);
    LastBr->setSuccessor(0, NextBB);
    LastBr = NextIRB.CreateCondBr(ShadowsEq, FallbackBB, FallbackBB);
  }

  LastBr->setSuccessor(0, Tail);
  FallbackIRB.CreateBr(Tail);
  PHINode *Shadow =
      PHINode::Create(DFS.PrimitiveShadowTy, 2, "", &Tail->front());
  Shadow->addIncoming(FallbackCall, FallbackBB);
  Shadow->addIncoming(TruncShadow, LastBr->getParent());
  return Shadow;
}

// Generates IR to load shadow corresponding to bytes [Addr, Addr+Size), where
// Addr has alignment Align, and take the union of each of those shadows. The
// returned shadow always has primitive type.
Value *DFSanFunction::loadShadow(Value *Addr, uint64_t Size, uint64_t Align,
                                 Instruction *Pos) {
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Addr)) {
    const auto i = AllocaShadowMap.find(AI);
    if (i != AllocaShadowMap.end()) {
      IRBuilder<> IRB(Pos);
      return IRB.CreateLoad(DFS.PrimitiveShadowTy, i->second);
    }
  }

  const llvm::Align ShadowAlign(Align * DFS.ShadowWidthBytes);
  SmallVector<const Value *, 2> Objs;
  getUnderlyingObjects(Addr, Objs);
  bool AllConstants = true;
  for (const Value *Obj : Objs) {
    if (isa<Function>(Obj) || isa<BlockAddress>(Obj))
      continue;
    if (isa<GlobalVariable>(Obj) && cast<GlobalVariable>(Obj)->isConstant())
      continue;

    AllConstants = false;
    break;
  }
  if (AllConstants)
    return DFS.ZeroPrimitiveShadow;

  Value *ShadowAddr = DFS.getShadowAddress(Addr, Pos);
  switch (Size) {
  case 0:
    return DFS.ZeroPrimitiveShadow;
  case 1: {
    LoadInst *LI = new LoadInst(DFS.PrimitiveShadowTy, ShadowAddr, "", Pos);
    LI->setAlignment(ShadowAlign);
    return LI;
  }
  case 2: {
    IRBuilder<> IRB(Pos);
    Value *ShadowAddr1 = IRB.CreateGEP(DFS.PrimitiveShadowTy, ShadowAddr,
                                       ConstantInt::get(DFS.IntptrTy, 1));
    return combineShadows(
        IRB.CreateAlignedLoad(DFS.PrimitiveShadowTy, ShadowAddr, ShadowAlign),
        IRB.CreateAlignedLoad(DFS.PrimitiveShadowTy, ShadowAddr1, ShadowAlign),
        Pos);
  }
  }

  if (ClFast16Labels && Size % (64 / DFS.ShadowWidthBits) == 0)
    return loadFast16ShadowFast(ShadowAddr, Size, ShadowAlign, Pos);

  if (!AvoidNewBlocks && Size % (64 / DFS.ShadowWidthBits) == 0)
    return loadLegacyShadowFast(ShadowAddr, Size, ShadowAlign, Pos);

  IRBuilder<> IRB(Pos);
  FunctionCallee &UnionLoadFn =
      ClFast16Labels ? DFS.DFSanUnionLoadFast16LabelsFn : DFS.DFSanUnionLoadFn;
  CallInst *FallbackCall = IRB.CreateCall(
      UnionLoadFn, {ShadowAddr, ConstantInt::get(DFS.IntptrTy, Size)});
  FallbackCall->addAttribute(AttributeList::ReturnIndex, Attribute::ZExt);
  return FallbackCall;
}

void DFSanVisitor::visitLoadInst(LoadInst &LI) {
  auto &DL = LI.getModule()->getDataLayout();
  uint64_t Size = DL.getTypeStoreSize(LI.getType());
  if (Size == 0) {
    DFSF.setShadow(&LI, DFSF.DFS.getZeroShadow(&LI));
    return;
  }

  Align Alignment = ClPreserveAlignment ? LI.getAlign() : Align(1);
  Value *PrimitiveShadow =
      DFSF.loadShadow(LI.getPointerOperand(), Size, Alignment.value(), &LI);
  if (ClCombinePointerLabelsOnLoad) {
    Value *PtrShadow = DFSF.getShadow(LI.getPointerOperand());
    PrimitiveShadow = DFSF.combineShadows(PrimitiveShadow, PtrShadow, &LI);
  }
  if (!DFSF.DFS.isZeroShadow(PrimitiveShadow))
    DFSF.NonZeroChecks.push_back(PrimitiveShadow);

  Value *Shadow =
      DFSF.expandFromPrimitiveShadow(LI.getType(), PrimitiveShadow, &LI);
  DFSF.setShadow(&LI, Shadow);
  if (ClEventCallbacks) {
    IRBuilder<> IRB(&LI);
    Value *Addr8 = IRB.CreateBitCast(LI.getPointerOperand(), DFSF.DFS.Int8Ptr);
    IRB.CreateCall(DFSF.DFS.DFSanLoadCallbackFn, {PrimitiveShadow, Addr8});
  }
}

void DFSanFunction::storePrimitiveShadow(Value *Addr, uint64_t Size,
                                         Align Alignment,
                                         Value *PrimitiveShadow,
                                         Instruction *Pos) {
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Addr)) {
    const auto i = AllocaShadowMap.find(AI);
    if (i != AllocaShadowMap.end()) {
      IRBuilder<> IRB(Pos);
      IRB.CreateStore(PrimitiveShadow, i->second);
      return;
    }
  }

  const Align ShadowAlign(Alignment.value() * DFS.ShadowWidthBytes);
  IRBuilder<> IRB(Pos);
  Value *ShadowAddr = DFS.getShadowAddress(Addr, Pos);
  if (DFS.isZeroShadow(PrimitiveShadow)) {
    IntegerType *ShadowTy =
        IntegerType::get(*DFS.Ctx, Size * DFS.ShadowWidthBits);
    Value *ExtZeroShadow = ConstantInt::get(ShadowTy, 0);
    Value *ExtShadowAddr =
        IRB.CreateBitCast(ShadowAddr, PointerType::getUnqual(ShadowTy));
    IRB.CreateAlignedStore(ExtZeroShadow, ExtShadowAddr, ShadowAlign);
    return;
  }

  const unsigned ShadowVecSize = 128 / DFS.ShadowWidthBits;
  uint64_t Offset = 0;
  if (Size >= ShadowVecSize) {
    auto *ShadowVecTy =
        FixedVectorType::get(DFS.PrimitiveShadowTy, ShadowVecSize);
    Value *ShadowVec = UndefValue::get(ShadowVecTy);
    for (unsigned i = 0; i != ShadowVecSize; ++i) {
      ShadowVec = IRB.CreateInsertElement(
          ShadowVec, PrimitiveShadow,
          ConstantInt::get(Type::getInt32Ty(*DFS.Ctx), i));
    }
    Value *ShadowVecAddr =
        IRB.CreateBitCast(ShadowAddr, PointerType::getUnqual(ShadowVecTy));
    do {
      Value *CurShadowVecAddr =
          IRB.CreateConstGEP1_32(ShadowVecTy, ShadowVecAddr, Offset);
      IRB.CreateAlignedStore(ShadowVec, CurShadowVecAddr, ShadowAlign);
      Size -= ShadowVecSize;
      ++Offset;
    } while (Size >= ShadowVecSize);
    Offset *= ShadowVecSize;
  }
  while (Size > 0) {
    Value *CurShadowAddr =
        IRB.CreateConstGEP1_32(DFS.PrimitiveShadowTy, ShadowAddr, Offset);
    IRB.CreateAlignedStore(PrimitiveShadow, CurShadowAddr, ShadowAlign);
    --Size;
    ++Offset;
  }
}

void DFSanVisitor::visitStoreInst(StoreInst &SI) {
  auto &DL = SI.getModule()->getDataLayout();
  uint64_t Size = DL.getTypeStoreSize(SI.getValueOperand()->getType());
  if (Size == 0)
    return;

  const Align Alignment = ClPreserveAlignment ? SI.getAlign() : Align(1);

  Value* Shadow = DFSF.getShadow(SI.getValueOperand());
  Value *PrimitiveShadow;
  if (ClCombinePointerLabelsOnStore) {
    Value *PtrShadow = DFSF.getShadow(SI.getPointerOperand());
    PrimitiveShadow = DFSF.combineShadows(Shadow, PtrShadow, &SI);
  } else {
    PrimitiveShadow = DFSF.collapseToPrimitiveShadow(Shadow, &SI);
  }
  DFSF.storePrimitiveShadow(SI.getPointerOperand(), Size, Alignment,
                            PrimitiveShadow, &SI);
  if (ClEventCallbacks) {
    IRBuilder<> IRB(&SI);
    Value *Addr8 = IRB.CreateBitCast(SI.getPointerOperand(), DFSF.DFS.Int8Ptr);
    IRB.CreateCall(DFSF.DFS.DFSanStoreCallbackFn, {PrimitiveShadow, Addr8});
  }
}

void DFSanVisitor::visitUnaryOperator(UnaryOperator &UO) {
  visitInstOperands(UO);
}

void DFSanVisitor::visitBinaryOperator(BinaryOperator &BO) {
  visitInstOperands(BO);
}

void DFSanVisitor::visitCastInst(CastInst &CI) { visitInstOperands(CI); }

void DFSanVisitor::visitCmpInst(CmpInst &CI) {
  visitInstOperands(CI);
  if (ClEventCallbacks) {
    IRBuilder<> IRB(&CI);
    Value *CombinedShadow = DFSF.getShadow(&CI);
    IRB.CreateCall(DFSF.DFS.DFSanCmpCallbackFn, CombinedShadow);
  }
}

void DFSanVisitor::visitGetElementPtrInst(GetElementPtrInst &GEPI) {
  visitInstOperands(GEPI);
}

void DFSanVisitor::visitExtractElementInst(ExtractElementInst &I) {
  visitInstOperands(I);
}

void DFSanVisitor::visitInsertElementInst(InsertElementInst &I) {
  visitInstOperands(I);
}

void DFSanVisitor::visitShuffleVectorInst(ShuffleVectorInst &I) {
  visitInstOperands(I);
}

void DFSanVisitor::visitExtractValueInst(ExtractValueInst &I) {
  if (!DFSF.DFS.shouldTrackFieldsAndIndices()) {
    visitInstOperands(I);
    return;
  }

  IRBuilder<> IRB(&I);
  Value *Agg = I.getAggregateOperand();
  Value *AggShadow = DFSF.getShadow(Agg);
  Value *ResShadow = IRB.CreateExtractValue(AggShadow, I.getIndices());
  DFSF.setShadow(&I, ResShadow);
  visitInstOperandOrigins(I);
}

void DFSanVisitor::visitInsertValueInst(InsertValueInst &I) {
  if (!DFSF.DFS.shouldTrackFieldsAndIndices()) {
    visitInstOperands(I);
    return;
  }

  IRBuilder<> IRB(&I);
  Value *AggShadow = DFSF.getShadow(I.getAggregateOperand());
  Value *InsShadow = DFSF.getShadow(I.getInsertedValueOperand());
  Value *Res = IRB.CreateInsertValue(AggShadow, InsShadow, I.getIndices());
  DFSF.setShadow(&I, Res);
  visitInstOperandOrigins(I);
}

void DFSanVisitor::visitAllocaInst(AllocaInst &I) {
  bool AllLoadsStores = true;
  for (User *U : I.users()) {
    if (isa<LoadInst>(U))
      continue;

    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() == &I)
        continue;
    }

    AllLoadsStores = false;
    break;
  }
  if (AllLoadsStores) {
    IRBuilder<> IRB(&I);
    DFSF.AllocaShadowMap[&I] = IRB.CreateAlloca(DFSF.DFS.PrimitiveShadowTy);
  }
  DFSF.setShadow(&I, DFSF.DFS.ZeroPrimitiveShadow);
}

void DFSanVisitor::visitSelectInst(SelectInst &I) {
  Value *CondShadow = DFSF.getShadow(I.getCondition());
  Value *TrueShadow = DFSF.getShadow(I.getTrueValue());
  Value *FalseShadow = DFSF.getShadow(I.getFalseValue());
  Value *ShadowSel = nullptr;
  const bool ShouldTrackOrigins = DFSF.DFS.shouldTrackOrigins();
  std::vector<Value *> Shadows;
  std::vector<Value *> Origins;
  Value *TrueOrigin =
      ShouldTrackOrigins ? DFSF.getOrigin(I.getTrueValue()) : nullptr;
  Value *FalseOrigin =
      ShouldTrackOrigins ? DFSF.getOrigin(I.getFalseValue()) : nullptr;

  if (isa<VectorType>(I.getCondition()->getType())) {
    ShadowSel = DFSF.combineShadowsThenConvert(I.getType(), TrueShadow,
                                               FalseShadow, &I);
    if (ShouldTrackOrigins) {
      Shadows.push_back(TrueShadow);
      Shadows.push_back(FalseShadow);
      Origins.push_back(TrueOrigin);
      Origins.push_back(FalseOrigin);
    }
  } else {
    if (TrueShadow == FalseShadow) {
      ShadowSel = TrueShadow;
      if (ShouldTrackOrigins) {
        Shadows.push_back(TrueShadow);
        Origins.push_back(TrueOrigin);
      }
    } else {
      ShadowSel =
          SelectInst::Create(I.getCondition(), TrueShadow, FalseShadow, "", &I);
      if (ShouldTrackOrigins) {
        Shadows.push_back(ShadowSel);
        Origins.push_back(SelectInst::Create(I.getCondition(), TrueOrigin,
                                             FalseOrigin, "", &I));
      }
    }
  }
  DFSF.setShadow(&I, ClTrackSelectControlFlow
                         ? DFSF.combineShadowsThenConvert(
                               I.getType(), CondShadow, ShadowSel, &I)
                         : ShadowSel);
  if (ShouldTrackOrigins) {
    if (ClTrackSelectControlFlow) {
      Shadows.push_back(CondShadow);
      Origins.push_back(DFSF.getOrigin(I.getCondition()));
    }
    DFSF.setOrigin(&I, DFSF.combineOrigins(Shadows, Origins, &I));
  }
}

void DFSanVisitor::visitMemSetInst(MemSetInst &I) {
  IRBuilder<> IRB(&I);
  Value *ValShadow = DFSF.getShadow(I.getValue());
  IRB.CreateCall(DFSF.DFS.DFSanSetLabelFn,
                 {ValShadow, IRB.CreateBitCast(I.getDest(), Type::getInt8PtrTy(
                                                                *DFSF.DFS.Ctx)),
                  IRB.CreateZExtOrTrunc(I.getLength(), DFSF.DFS.IntptrTy)});
}

void DFSanVisitor::visitMemTransferInst(MemTransferInst &I) {
  IRBuilder<> IRB(&I);
  Value *RawDestShadow = DFSF.DFS.getShadowAddress(I.getDest(), &I);
  Value *SrcShadow = DFSF.DFS.getShadowAddress(I.getSource(), &I);
  Value *LenShadow =
      IRB.CreateMul(I.getLength(), ConstantInt::get(I.getLength()->getType(),
                                                    DFSF.DFS.ShadowWidthBytes));
  Type *Int8Ptr = Type::getInt8PtrTy(*DFSF.DFS.Ctx);
  Value *DestShadow = IRB.CreateBitCast(RawDestShadow, Int8Ptr);
  SrcShadow = IRB.CreateBitCast(SrcShadow, Int8Ptr);
  auto *MTI = cast<MemTransferInst>(
      IRB.CreateCall(I.getFunctionType(), I.getCalledOperand(),
                     {DestShadow, SrcShadow, LenShadow, I.getVolatileCst()}));
  if (ClPreserveAlignment) {
    MTI->setDestAlignment(I.getDestAlign() * DFSF.DFS.ShadowWidthBytes);
    MTI->setSourceAlignment(I.getSourceAlign() * DFSF.DFS.ShadowWidthBytes);
  } else {
    MTI->setDestAlignment(Align(DFSF.DFS.ShadowWidthBytes));
    MTI->setSourceAlignment(Align(DFSF.DFS.ShadowWidthBytes));
  }
  if (ClEventCallbacks) {
    IRB.CreateCall(DFSF.DFS.DFSanMemTransferCallbackFn,
                   {RawDestShadow, I.getLength()});
  }
}

void DFSanVisitor::visitReturnInst(ReturnInst &RI) {
  if (!DFSF.IsNativeABI && RI.getReturnValue()) {
    switch (DFSF.IA) {
    case DataFlowSanitizer::IA_TLS: {
      Value *S = DFSF.getShadow(RI.getReturnValue());
      IRBuilder<> IRB(&RI);
      Type *RT = DFSF.F->getFunctionType()->getReturnType();
      unsigned Size =
          getDataLayout().getTypeAllocSize(DFSF.DFS.getShadowTy(RT));
      if (Size <= kRetvalTLSSize) {
        // If the size overflows, stores nothing. At callsite, oversized return
        // shadows are set to zero.
        IRB.CreateAlignedStore(S, DFSF.getRetvalTLS(RT, IRB),
                               kShadowTLSAlignment);
      }
      if (DFSF.DFS.shouldTrackOrigins()) {
        Value *O = DFSF.getOrigin(RI.getReturnValue());
        IRB.CreateStore(O, DFSF.getRetvalOriginTLS());
      }
      break;
    }
    case DataFlowSanitizer::IA_Args: {
      IRBuilder<> IRB(&RI);
      Type *RT = DFSF.F->getFunctionType()->getReturnType();
      Value *InsVal =
          IRB.CreateInsertValue(UndefValue::get(RT), RI.getReturnValue(), 0);
      Value *InsShadow =
          IRB.CreateInsertValue(InsVal, DFSF.getShadow(RI.getReturnValue()), 1);
      RI.setOperand(0, InsShadow);
      break;
    }
    }
  }
}

bool DFSanVisitor::visitWrappedCallBase(Function &F, CallBase &CB) {
  IRBuilder<> IRB(&CB);
  switch (DFSF.DFS.getWrapperKind(&F)) {
  case DataFlowSanitizer::WK_Warning:
    CB.setCalledFunction(&F);
    IRB.CreateCall(DFSF.DFS.DFSanUnimplementedFn,
                   IRB.CreateGlobalStringPtr(F.getName()));
    DFSF.setShadow(&CB, DFSF.DFS.getZeroShadow(&CB));
    return true;
  case DataFlowSanitizer::WK_Discard:
    CB.setCalledFunction(&F);
    DFSF.setShadow(&CB, DFSF.DFS.getZeroShadow(&CB));
    return true;
  case DataFlowSanitizer::WK_Functional:
    CB.setCalledFunction(&F);
    visitInstOperands(CB);
    return true;
  case DataFlowSanitizer::WK_Custom:
    // Don't try to handle invokes of custom functions, it's too complicated.
    // Instead, invoke the dfsw$ wrapper, which will in turn call the __dfsw_
    // wrapper.
    CallInst *CI = dyn_cast<CallInst>(&CB);
    if (!CI)
      return false;

    FunctionType *FT = F.getFunctionType();
    TransformedFunction CustomFn = DFSF.DFS.getCustomFunctionType(FT);
    std::string CustomFName = "__dfsw_";
    CustomFName += F.getName();
    FunctionCallee CustomF = DFSF.DFS.Mod->getOrInsertFunction(
        CustomFName, CustomFn.TransformedType);
    if (Function *CustomFn = dyn_cast<Function>(CustomF.getCallee())) {
      CustomFn->copyAttributesFrom(&F);

      // Custom functions returning non-void will write to the return label.
      if (!FT->getReturnType()->isVoidTy()) {
        CustomFn->removeAttributes(AttributeList::FunctionIndex,
                                   DFSF.DFS.ReadOnlyNoneAttrs);
      }
    }

    std::vector<Value *> Args;

    // Adds non-variable arguments.
    auto *I = CB.arg_begin();
    for (unsigned n = FT->getNumParams(); n != 0; ++I, --n) {
      Type *T = (*I)->getType();
      FunctionType *ParamFT;
      if (isa<PointerType>(T) &&
          (ParamFT = dyn_cast<FunctionType>(
               cast<PointerType>(T)->getElementType()))) {
        std::string TName = "dfst";
        TName += utostr(FT->getNumParams() - n);
        TName += "$";
        TName += F.getName();
        Constant *T = DFSF.DFS.getOrBuildTrampolineFunction(ParamFT, TName);
        Args.push_back(T);
        Args.push_back(
            IRB.CreateBitCast(*I, Type::getInt8PtrTy(*DFSF.DFS.Ctx)));
      } else {
        Args.push_back(*I);
      }
    }

    // Adds non-variable argument shadows.
    I = CB.arg_begin();
    const unsigned ShadowArgStart = Args.size();
    for (unsigned N = FT->getNumParams(); N != 0; ++I, --N)
      Args.push_back(DFSF.collapseToPrimitiveShadow(DFSF.getShadow(*I), &CB));

    // Adds variable argument shadows.
    if (FT->isVarArg()) {
      auto *LabelVATy = ArrayType::get(DFSF.DFS.PrimitiveShadowTy,
                                       CB.arg_size() - FT->getNumParams());
      auto *LabelVAAlloca =
          new AllocaInst(LabelVATy, getDataLayout().getAllocaAddrSpace(),
                         "labelva", &DFSF.F->getEntryBlock().front());

      for (unsigned N = 0; I != CB.arg_end(); ++I, ++N) {
        auto *LabelVAPtr = IRB.CreateStructGEP(LabelVATy, LabelVAAlloca, N);
        IRB.CreateStore(DFSF.collapseToPrimitiveShadow(DFSF.getShadow(*I), &CB),
                        LabelVAPtr);
      }

      Args.push_back(IRB.CreateStructGEP(LabelVATy, LabelVAAlloca, 0));
    }

    // Adds the return value shadow.
    if (!FT->getReturnType()->isVoidTy()) {
      if (!DFSF.LabelReturnAlloca) {
        DFSF.LabelReturnAlloca = new AllocaInst(
            DFSF.DFS.PrimitiveShadowTy, getDataLayout().getAllocaAddrSpace(),
            "labelreturn", &DFSF.F->getEntryBlock().front());
      }
      Args.push_back(DFSF.LabelReturnAlloca);
    }

    // Adds variable arguments.
    append_range(Args, drop_begin(CB.args(), FT->getNumParams()));

    CallInst *CustomCI = IRB.CreateCall(CustomF, Args);
    CustomCI->setCallingConv(CI->getCallingConv());
    CustomCI->setAttributes(TransformFunctionAttributes(
        CustomFn, CI->getContext(), CI->getAttributes()));

    // Update the parameter attributes of the custom call instruction to
    // zero extend the shadow parameters. This is required for targets
    // which consider PrimitiveShadowTy an illegal type.
    for (unsigned N = 0; N < FT->getNumParams(); N++) {
      const unsigned ArgNo = ShadowArgStart + N;
      if (CustomCI->getArgOperand(ArgNo)->getType() ==
          DFSF.DFS.PrimitiveShadowTy)
        CustomCI->addParamAttr(ArgNo, Attribute::ZExt);
    }

    // Loads the return value shadow.
    if (!FT->getReturnType()->isVoidTy()) {
      LoadInst *LabelLoad =
          IRB.CreateLoad(DFSF.DFS.PrimitiveShadowTy, DFSF.LabelReturnAlloca);
      DFSF.setShadow(CustomCI, DFSF.expandFromPrimitiveShadow(
                                   FT->getReturnType(), LabelLoad, &CB));
    }

    CI->replaceAllUsesWith(CustomCI);
    CI->eraseFromParent();
    return true;
  }
  return false;
}

void DFSanVisitor::visitCallBase(CallBase &CB) {
  Function *F = CB.getCalledFunction();
  if ((F && F->isIntrinsic()) || CB.isInlineAsm()) {
    visitInstOperands(CB);
    return;
  }

  // Calls to this function are synthesized in wrappers, and we shouldn't
  // instrument them.
  if (F == DFSF.DFS.DFSanVarargWrapperFn.getCallee()->stripPointerCasts())
    return;

  DenseMap<Value *, Function *>::iterator i =
      DFSF.DFS.UnwrappedFnMap.find(CB.getCalledOperand());
  if (i != DFSF.DFS.UnwrappedFnMap.end())
    if (visitWrappedCallBase(*i->second, CB))
      return;

  IRBuilder<> IRB(&CB);

  FunctionType *FT = CB.getFunctionType();
  if (DFSF.DFS.getInstrumentedABI() == DataFlowSanitizer::IA_TLS) {
    // Stores argument shadows.
    unsigned ArgOffset = 0;
    const DataLayout &DL = getDataLayout();
    for (unsigned I = 0, N = FT->getNumParams(); I != N; ++I) {
      unsigned Size =
          DL.getTypeAllocSize(DFSF.DFS.getShadowTy(FT->getParamType(I)));
      // Stop storing if arguments' size overflows. Inside a function, arguments
      // after overflow have zero shadow values.
      if (ArgOffset + Size > kArgTLSSize)
        break;
      IRB.CreateAlignedStore(
          DFSF.getShadow(CB.getArgOperand(I)),
          DFSF.getArgTLS(FT->getParamType(I), ArgOffset, IRB),
          kShadowTLSAlignment);
      ArgOffset += alignTo(Size, kShadowTLSAlignment);
    }
  }

  Instruction *Next = nullptr;
  if (!CB.getType()->isVoidTy()) {
    if (InvokeInst *II = dyn_cast<InvokeInst>(&CB)) {
      if (II->getNormalDest()->getSinglePredecessor()) {
        Next = &II->getNormalDest()->front();
      } else {
        BasicBlock *NewBB =
            SplitEdge(II->getParent(), II->getNormalDest(), &DFSF.DT);
        Next = &NewBB->front();
      }
    } else {
      assert(CB.getIterator() != CB.getParent()->end());
      Next = CB.getNextNode();
    }

    if (DFSF.DFS.getInstrumentedABI() == DataFlowSanitizer::IA_TLS) {
      // Loads the return value shadow.
      IRBuilder<> NextIRB(Next);
      const DataLayout &DL = getDataLayout();
      unsigned Size = DL.getTypeAllocSize(DFSF.DFS.getShadowTy(&CB));
      if (Size > kRetvalTLSSize) {
        // Set overflowed return shadow to be zero.
        DFSF.setShadow(&CB, DFSF.DFS.getZeroShadow(&CB));
      } else {
        LoadInst *LI = NextIRB.CreateAlignedLoad(
            DFSF.DFS.getShadowTy(&CB), DFSF.getRetvalTLS(CB.getType(), NextIRB),
            kShadowTLSAlignment, "_dfsret");
        DFSF.SkipInsts.insert(LI);
        DFSF.setShadow(&CB, LI);
        DFSF.NonZeroChecks.push_back(LI);
      }
    }
  }

  // Do all instrumentation for IA_Args down here to defer tampering with the
  // CFG in a way that SplitEdge may be able to detect.
  if (DFSF.DFS.getInstrumentedABI() == DataFlowSanitizer::IA_Args) {
    FunctionType *NewFT = DFSF.DFS.getArgsFunctionType(FT);
    Value *Func =
        IRB.CreateBitCast(CB.getCalledOperand(), PointerType::getUnqual(NewFT));
    std::vector<Value *> Args;

    auto i = CB.arg_begin(), E = CB.arg_end();
    for (unsigned n = FT->getNumParams(); n != 0; ++i, --n)
      Args.push_back(*i);

    i = CB.arg_begin();
    for (unsigned n = FT->getNumParams(); n != 0; ++i, --n)
      Args.push_back(DFSF.getShadow(*i));

    if (FT->isVarArg()) {
      unsigned VarArgSize = CB.arg_size() - FT->getNumParams();
      ArrayType *VarArgArrayTy =
          ArrayType::get(DFSF.DFS.PrimitiveShadowTy, VarArgSize);
      AllocaInst *VarArgShadow =
        new AllocaInst(VarArgArrayTy, getDataLayout().getAllocaAddrSpace(),
                       "", &DFSF.F->getEntryBlock().front());
      Args.push_back(IRB.CreateConstGEP2_32(VarArgArrayTy, VarArgShadow, 0, 0));
      for (unsigned n = 0; i != E; ++i, ++n) {
        IRB.CreateStore(
            DFSF.getShadow(*i),
            IRB.CreateConstGEP2_32(VarArgArrayTy, VarArgShadow, 0, n));
        Args.push_back(*i);
      }
    }

    CallBase *NewCB;
    if (InvokeInst *II = dyn_cast<InvokeInst>(&CB)) {
      NewCB = IRB.CreateInvoke(NewFT, Func, II->getNormalDest(),
                               II->getUnwindDest(), Args);
    } else {
      NewCB = IRB.CreateCall(NewFT, Func, Args);
    }
    NewCB->setCallingConv(CB.getCallingConv());
    NewCB->setAttributes(CB.getAttributes().removeAttributes(
        *DFSF.DFS.Ctx, AttributeList::ReturnIndex,
        AttributeFuncs::typeIncompatible(NewCB->getType())));

    if (Next) {
      ExtractValueInst *ExVal = ExtractValueInst::Create(NewCB, 0, "", Next);
      DFSF.SkipInsts.insert(ExVal);
      ExtractValueInst *ExShadow = ExtractValueInst::Create(NewCB, 1, "", Next);
      DFSF.SkipInsts.insert(ExShadow);
      DFSF.setShadow(ExVal, ExShadow);
      DFSF.NonZeroChecks.push_back(ExShadow);

      CB.replaceAllUsesWith(ExVal);
    }

    CB.eraseFromParent();
  }
}

void DFSanVisitor::visitPHINode(PHINode &PN) {
  Type *ShadowTy = DFSF.DFS.getShadowTy(&PN);
  PHINode *ShadowPN =
      PHINode::Create(ShadowTy, PN.getNumIncomingValues(), "", &PN);

  // Give the shadow phi node valid predecessors to fool SplitEdge into working.
  Value *UndefShadow = UndefValue::get(ShadowTy);
  for (PHINode::block_iterator i = PN.block_begin(), e = PN.block_end(); i != e;
       ++i) {
    ShadowPN->addIncoming(UndefShadow, *i);
  }

  DFSF.PHIFixups.push_back(std::make_pair(&PN, ShadowPN));
  DFSF.setShadow(&PN, ShadowPN);
}

namespace {
class DataFlowSanitizerLegacyPass : public ModulePass {
private:
  std::vector<std::string> ABIListFiles;

public:
  static char ID;

  DataFlowSanitizerLegacyPass(
      const std::vector<std::string> &ABIListFiles = std::vector<std::string>())
      : ModulePass(ID), ABIListFiles(ABIListFiles) {}

  bool runOnModule(Module &M) override {
    return DataFlowSanitizer(ABIListFiles).runImpl(M);
  }
};
} // namespace

char DataFlowSanitizerLegacyPass::ID;

INITIALIZE_PASS(DataFlowSanitizerLegacyPass, "dfsan",
                "DataFlowSanitizer: dynamic data flow analysis.", false, false)

ModulePass *llvm::createDataFlowSanitizerLegacyPassPass(
    const std::vector<std::string> &ABIListFiles) {
  return new DataFlowSanitizerLegacyPass(ABIListFiles);
}

PreservedAnalyses DataFlowSanitizerPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  if (DataFlowSanitizer(ABIListFiles).runImpl(M)) {
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}
