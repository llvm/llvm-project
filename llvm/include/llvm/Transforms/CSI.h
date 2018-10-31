//===-- CSI.h -- CSI implementation structures and hooks -----*- C++ -*----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of CSI, a framework that provides comprehensive static
// instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CSI_H
#define LLVM_TRANSFORMS_CSI_H

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {

static const char *const CsiRtUnitInitName = "__csirt_unit_init";
static const char *const CsiRtUnitCtorName = "csirt.unit_ctor";
static const char *const CsiFunctionBaseIdName = "__csi_unit_func_base_id";
static const char *const CsiFunctionExitBaseIdName = "__csi_unit_func_exit_base_id";
static const char *const CsiBasicBlockBaseIdName = "__csi_unit_bb_base_id";
static const char *const CsiCallsiteBaseIdName = "__csi_unit_callsite_base_id";
static const char *const CsiLoadBaseIdName = "__csi_unit_load_base_id";
static const char *const CsiStoreBaseIdName = "__csi_unit_store_base_id";
static const char *const CsiAllocaBaseIdName = "__csi_unit_alloca_base_id";
static const char *const CsiDetachBaseIdName = "__csi_unit_detach_base_id";
static const char *const CsiTaskBaseIdName = "__csi_unit_task_base_id";
static const char *const CsiTaskExitBaseIdName =
  "__csi_unit_task_exit_base_id";
static const char *const CsiDetachContinueBaseIdName =
  "__csi_unit_detach_continue_base_id";
static const char *const CsiSyncBaseIdName = "__csi_unit_sync_base_id";
static const char *const CsiUnitSizeTableName = "__csi_unit_size_table";
static const char *const CsiUnitFedTableName = "__csi_unit_fed_table";
static const char *const CsiFuncIdVariablePrefix = "__csi_func_id_";
static const char *const CsiUnitFedTableArrayName = "__csi_unit_fed_tables";
static const char *const CsiUnitSizeTableArrayName = "__csi_unit_size_tables";
static const char *const CsiInitCallsiteToFunctionName =
    "__csi_init_callsite_to_function";
static const char *const CsiDisableInstrumentationName =
    "__csi_disable_instrumentation";

using csi_id_t = int64_t;
static const csi_id_t CsiUnknownId = -1;
static const csi_id_t CsiCallsiteUnknownTargetId = CsiUnknownId;
// See llvm/tools/clang/lib/CodeGen/CodeGenModule.h:
static const int CsiUnitCtorPriority = 65535;

/// Maintains a mapping from CSI ID to static data for that ID.
class ForensicTable {
public:
  ForensicTable() : BaseId(nullptr), IdCounter(0) {}
  ForensicTable(Module &M, StringRef BaseIdName);

  /// The number of entries in this forensic table
  uint64_t size() const { return IdCounter; }

  /// Get the local ID of the given Value.
  uint64_t getId(const Value *V);

  /// The GlobalVariable holding the base ID for this forensic table.
  GlobalVariable *baseId() const { return BaseId; }

  /// Converts a local to global ID conversion.
  ///
  /// This is done by using the given IRBuilder to insert a load to the base ID
  /// global variable followed by an add of the base value and the local ID.
  ///
  /// \returns A Value holding the global ID corresponding to the
  /// given local ID.
  Value *localToGlobalId(uint64_t LocalId, IRBuilder<> &IRB) const;

  /// Helper function to get or create a string for a forensic-table entry.
  static Constant *getObjectStrGV(Module &M, StringRef Str, const Twine GVName);

protected:
  /// The GlobalVariable holding the base ID for this FED table.
  GlobalVariable *BaseId;
  /// Counter of local IDs used so far.
  uint64_t IdCounter;
  /// Map of Value to Local ID.
  DenseMap<const Value *, uint64_t> ValueToLocalIdMap;
};

/// Maintains a mapping from CSI ID to front-end data for that ID.
///
/// The front-end data currently is the source location that a given
/// CSI ID corresponds to.
class FrontEndDataTable : public ForensicTable {
public:
  FrontEndDataTable() : ForensicTable() {}
  FrontEndDataTable(Module &M, StringRef BaseIdName)
      : ForensicTable(M, BaseIdName) {}

  /// The number of entries in this FED table
  uint64_t size() const { return LocalIdToSourceLocationMap.size(); }

  /// Add the given Function to this FED table.
  /// \returns The local ID of the Function.
  uint64_t add(const Function &F);

  /// Add the given BasicBlock to this FED table.
  /// \returns The local ID of the BasicBlock.
  uint64_t add(const BasicBlock &BB);

  /// Add the given Instruction to this FED table.
  /// \returns The local ID of the Instruction.
  uint64_t add(const Instruction &I);

  /// Get the Type for a pointer to a FED table entry.
  ///
  /// A FED table entry is just a source location.
  static PointerType *getPointerType(LLVMContext &C);

  /// Insert this FED table into the given Module.
  ///
  /// The FED table is constructed as a ConstantArray indexed by local
  /// IDs.  The runtime is responsible for performing the mapping that
  /// allows the table to be indexed by global ID.
  Constant *insertIntoModule(Module &M) const;

private:
  struct SourceLocation {
    StringRef Name;
    int32_t Line;
    int32_t Column;
    StringRef Filename;
    StringRef Directory;
  };

  /// Map of local ID to SourceLocation.
  DenseMap<uint64_t, SourceLocation> LocalIdToSourceLocationMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSourceLocStructType(LLVMContext &C);

  /// Append the debug information to the table, assigning it the next
  /// available ID.
  ///
  /// \returns The local ID of the appended information.
  /// @{
  void add(uint64_t ID, const DILocation *Loc);
  void add(uint64_t ID, const DISubprogram *Subprog);
  /// @}

  /// Append the line and file information to the table, assigning it
  /// the next available ID.
  ///
  /// \returns The new local ID of the DILocation.
  void add(uint64_t ID, int32_t Line = -1, int32_t Column = -1,
           StringRef Filename = "", StringRef Directory = "",
           StringRef Name = "");
};

/// Maintains a mapping from CSI ID of a basic block to the size of that basic
/// block in LLVM IR instructions.
class SizeTable : public ForensicTable {
public:
  SizeTable() : ForensicTable() {}
  SizeTable(Module &M, StringRef BaseIdName)
      : ForensicTable(M, BaseIdName) {}

  /// The number of entries in this table
  uint64_t size() const { return LocalIdToSizeMap.size(); }

  /// Add the given basic block  to this table.
  /// \returns The local ID of the basic block.
  uint64_t add(const BasicBlock &BB);

  /// Get the Type for a pointer to a table entry.
  ///
  /// A table entry is just a source location.
  static PointerType *getPointerType(LLVMContext &C);

  /// Insert this table into the given Module.
  ///
  /// The table is constructed as a ConstantArray indexed by local IDs.  The
  /// runtime is responsible for performing the mapping that allows the table to
  /// be indexed by global ID.
  Constant *insertIntoModule(Module &M) const;

private:
  struct SizeInformation {
    // This count includes every IR instruction.
    int32_t FullIRSize;
    // This count excludes IR instructions that don't lower to any real
    // instructions, e.g., PHI instructions, debug intrinsics, and lifetime
    // intrinsics.
    int32_t NonEmptyIRSize;
  };

  /// Map of local ID to size.
  DenseMap<uint64_t, SizeInformation> LocalIdToSizeMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSizeStructType(LLVMContext &C);

  /// Append the size information to the table.
  void add(uint64_t ID, int32_t FullIRSize = 0, int32_t NonEmptyIRSize = 0);
};

/// Represents a property value passed to hooks.
class CsiProperty {
public:
  CsiProperty() {}

  /// Return the coerced type of a property.
  ///
  /// TODO: Right now, this function simply returns a 64-bit integer.  Although
  /// this solution works for x86_64, it should be generalized to handle other
  /// architectures in the future.
  static Type *getCoercedType(LLVMContext &C, StructType *Ty) {
    // Must match the definition of property type in csi.h
    // return StructType::get(IntegerType::get(C, 64),
    //                        nullptr);
    // We return an integer type, rather than a struct type, to deal with x86_64
    // type coercion on struct bit fields.
    return IntegerType::get(C, 64);
  }

  /// Return a constant value holding this property.
  virtual Constant *getValueImpl(LLVMContext &C) const = 0;

  Constant *getValue(IRBuilder<> &IRB) const {
    return getValueImpl(IRB.getContext());
  }
};

class CsiFuncProperty : public CsiProperty {
public:
  CsiFuncProperty() {
    PropValue.Bits = 0;
  }

  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.MaySpawn),
                           IntegerType::get(C, PropBits.Padding)));
  }
  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(StructTy,
    //                            ConstantInt::get(IntegerType::get(C, 64), 0),
    //                            nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the MaySpawn property.
  void setMaySpawn(bool v) {
    PropValue.Fields.MaySpawn = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned MaySpawn : 1;
      uint64_t Padding : 63;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int MaySpawn;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 1, (64-1) };
};

class CsiFuncExitProperty : public CsiProperty {
public:
  CsiFuncExitProperty() {
      PropValue.Bits = 0;
  }

  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.MaySpawn),
                           IntegerType::get(C, PropBits.EHReturn),
                           IntegerType::get(C, PropBits.Padding)));
  }
  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(StructTy,
    //                            ConstantInt::get(IntegerType::get(C, 64), 0),
    //                            nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the MaySpawn property.
  void setMaySpawn(bool v) {
    PropValue.Fields.MaySpawn = v;
  }

  /// Set the value of the EHReturn property.
  void setEHReturn(bool v) {
    PropValue.Fields.EHReturn = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned MaySpawn : 1;
      unsigned EHReturn : 1;
      uint64_t Padding : 62;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int MaySpawn;
    int EHReturn;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 1, 1, (64-1-1) };
};

class CsiBBProperty : public CsiProperty {
public:
  CsiBBProperty() {
    PropValue.Bits = 0;
  }

  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.IsLandingPad),
                           IntegerType::get(C, PropBits.IsEHPad),
                           IntegerType::get(C, PropBits.Padding)));
  }

  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(StructTy,
    //                            ConstantInt::get(IntegerType::get(C, 64), 0),
    //                            nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the IsLandingPad property.
  void setIsLandingPad(bool v) {
    PropValue.Fields.IsLandingPad = v;
  }

  /// Set the value of the IsEHPad property.
  void setIsEHPad(bool v) {
    PropValue.Fields.IsEHPad = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned IsLandingPad : 1;
      unsigned IsEHPad : 1;
      uint64_t Padding : 62;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int IsLandingPad;
    int IsEHPad;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 1, 1, (64-1-1) };
};

class CsiCallProperty : public CsiProperty {
public:
  CsiCallProperty() {
    PropValue.Bits = 0;
  }

  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.IsIndirect),
                           IntegerType::get(C, PropBits.Padding)));
  }
  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // StructType *StructTy = getType(C);
    // return ConstantStruct::get(
    //     StructTy,
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsIndirect),
    //                      PropValue.IsIndirect),
    //     ConstantInt::get(IntegerType::get(C, PropBits.Padding), 0),
    //     nullptr);
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the IsIndirect property.
  void setIsIndirect(bool v) {
    PropValue.Fields.IsIndirect = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned IsIndirect : 1;
      uint64_t Padding : 63;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int IsIndirect;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 1, (64-1) };
};

class CsiLoadStoreProperty : public CsiProperty {
public:
  CsiLoadStoreProperty() {
    PropValue.Bits = 0;
  }
  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.Alignment),
                           IntegerType::get(C, PropBits.IsVtableAccess),
                           IntegerType::get(C, PropBits.IsConstant),
                           IntegerType::get(C, PropBits.IsOnStack),
                           IntegerType::get(C, PropBits.MayBeCaptured),
                           IntegerType::get(C, PropBits.LoadReadBeforeWriteInBB),
                           IntegerType::get(C, PropBits.Padding)));
  }
  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // return ConstantStruct::get(
    //     StructTy,
    //     ConstantInt::get(IntegerType::get(C, PropBits.Alignment),
    //                      PropValue.Alignment),
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsVtableAccess),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsConstant),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C, PropBits.IsOnStack),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C, PropBits.MayBeCaptured),
    //                      PropValue.IsVtableAccess),
    //     ConstantInt::get(IntegerType::get(C, PropBits.LoadReadBeforeWriteInBB),
    //                      PropValue.LoadReadBeforeWriteInBB),
    //     ConstantInt::get(IntegerType::get(C, PropBits.Padding), 0),
    //     nullptr);
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the Alignment property.
  void setAlignment(char v) {
    PropValue.Fields.Alignment = v;
  }
  /// Set the value of the IsVtableAccess property.
  void setIsVtableAccess(bool v) {
    PropValue.Fields.IsVtableAccess = v;
  }
  /// Set the value of the IsConstant property.
  void setIsConstant(bool v) {
    PropValue.Fields.IsConstant = v;
  }
  /// Set the value of the IsOnStack property.
  void setIsOnStack(bool v) {
    PropValue.Fields.IsOnStack = v;
  }
  /// Set the value of the MayBeCaptured property.
  void setMayBeCaptured(bool v) {
    PropValue.Fields.MayBeCaptured = v;
  }
  /// Set the value of the LoadReadBeforeWriteInBB property.
  void setLoadReadBeforeWriteInBB(bool v) {
    PropValue.Fields.LoadReadBeforeWriteInBB = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned Alignment : 8;
      unsigned IsVtableAccess : 1;
      unsigned IsConstant : 1;
      unsigned IsOnStack : 1;
      unsigned MayBeCaptured : 1;
      unsigned LoadReadBeforeWriteInBB : 1;
      uint64_t Padding : 53;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int Alignment;
    int IsVtableAccess;
    int IsConstant;
    int IsOnStack;
    int MayBeCaptured;
    int LoadReadBeforeWriteInBB;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 8, 1, 1, 1, 1, 1, (64-8-1-1-1-1-1) };
};

class CsiAllocaProperty : public CsiProperty {
public:
  CsiAllocaProperty() {
    PropValue.Bits = 0;
  }

  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.IsStatic),
                           IntegerType::get(C, PropBits.Padding)));
  }
  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csi.h
    // TODO: This solution works for x86, but should be generalized to support
    // other architectures in the future.
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the IsIndirect property.
  void setIsStatic(bool v) {
    PropValue.Fields.IsStatic = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned IsStatic : 1;
      uint64_t Padding : 63;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int IsStatic;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 1, (64-1) };
};

struct CSIImpl {
public:
  CSIImpl(Module &M, CallGraph *CG,
          function_ref<DominatorTree &(Function &)> GetDomTree,
          const CSIOptions &Options = CSIOptions())
      : M(M), DL(M.getDataLayout()), CG(CG), GetDomTree(GetDomTree),
        Options(Options)
  {}

  bool run();

  /// Get the number of bytes accessed via the given address.
  static int getNumBytesAccessed(Value *Addr, const DataLayout &DL);

  /// Members to extract properties of loads/stores.
  static bool isVtableAccess(Instruction *I);
  static bool addrPointsToConstantData(Value *Addr);
  static bool isAtomic(Instruction *I);

  /// Helper functions to deal with calls to functions that can throw.
  static Constant *getDefaultPersonalityFn(Module *M);
  static void changeCallsToInvokes(Function &F);

  /// Helper function that identifies calls or invokes of placeholder functions,
  /// such as debug-info intrinsics or lifetime intrinsics.
  static bool callsPlaceholderFunction(const Instruction &I);

  static Constant *getDefaultID(IRBuilder<> &IRB) {
    return IRB.getInt64(CsiUnknownId);
  }

protected:
  /// Initialize the CSI pass.
  void initializeCsi();
  /// Finalize the CSI pass.
  void finalizeCsi();

  /// Initialize llvm::Functions for the CSI hooks.
  /// @{
  void initializeLoadStoreHooks();
  void initializeFuncHooks();
  void initializeBasicBlockHooks();
  void initializeCallsiteHooks();
  void initializeAllocaHooks();
  void initializeMemIntrinsicsHooks();
  void initializeTapirHooks();
  /// @}

  static StructType *getUnitFedTableType(LLVMContext &C,
                                         PointerType *EntryPointerType);
  static Constant *fedTableToUnitFedTable(Module &M,
                                          StructType *UnitFedTableType,
                                          FrontEndDataTable &FedTable);
  static StructType *getUnitSizeTableType(LLVMContext &C,
                                          PointerType *EntryPointerType);
  static Constant *sizeTableToUnitSizeTable(Module &M,
                                            StructType *UnitSizeTableType,
                                            SizeTable &SzTable);
  /// Initialize the front-end data table structures.
  void initializeFEDTables();
  /// Collect unit front-end data table structures for finalization.
  void collectUnitFEDTables();
  /// Initialize the front-end data table structures.
  void initializeSizeTables();
  /// Collect unit front-end data table structures for finalization.
  void collectUnitSizeTables();

  virtual CallInst *createRTUnitInitCall(IRBuilder<> &IRB);

  // Get the local ID of the given function.
  uint64_t getLocalFunctionID(Function &F);
  /// Generate a function that stores global function IDs into a set
  /// of externally-visible global variables.
  void generateInitCallsiteToFunction();

  /// Compute CSI properties on the given ordered list of loads and stores.
  void computeLoadAndStoreProperties(
      SmallVectorImpl<std::pair<Instruction *, CsiLoadStoreProperty>>
      &LoadAndStoreProperties,
      SmallVectorImpl<Instruction *> &BBLoadsAndStores,
      const DataLayout &DL);

  /// Insert calls to the instrumentation hooks.
  /// @{
  void addLoadStoreInstrumentation(Instruction *I, Function *BeforeFn,
                                   Function *AfterFn, Value *CsiId,
                                   Type *AddrType, Value *Addr, int NumBytes,
                                   CsiLoadStoreProperty &Prop);
  void instrumentLoadOrStore(Instruction *I, CsiLoadStoreProperty &Prop,
                             const DataLayout &DL);
  void instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  void instrumentCallsite(Instruction *I, DominatorTree *DT);
  void instrumentAlloca(Instruction *I);
  void instrumentBasicBlock(BasicBlock &BB);
  void instrumentDetach(DetachInst *DI, DominatorTree *DT);
  void instrumentSync(SyncInst *SI);
  void instrumentFunction(Function &F);
  /// @}

  /// Insert a conditional call to the given hook function before the
  /// given instruction. The condition is based on the value of
  /// __csi_disable_instrumentation.
  void insertConditionalHookCall(Instruction *I, Function *HookFunction,
                                 ArrayRef<Value *> HookArgs);

  /// Return true if the given function should not be instrumented.
  bool shouldNotInstrumentFunction(Function &F);

  Module &M;
  const DataLayout &DL;
  CallGraph *CG;
  function_ref<DominatorTree &(Function &)> GetDomTree;
  CSIOptions Options;

  FrontEndDataTable FunctionFED, FunctionExitFED, BasicBlockFED, CallsiteFED,
    LoadFED, StoreFED, 
    AllocaFED, 
    DetachFED, TaskFED, TaskExitFED, DetachContinueFED,
    SyncFED;

  SmallVector<Constant *, 12> UnitFedTables;

  SizeTable BBSize;
  SmallVector<Constant *, 1> UnitSizeTables;

  // Instrumentation hooks
  Function *CsiFuncEntry = nullptr, *CsiFuncExit = nullptr;
  Function *CsiBBEntry = nullptr, *CsiBBExit = nullptr;
  Function *CsiBeforeCallsite = nullptr, *CsiAfterCallsite = nullptr;
  Function *CsiBeforeRead = nullptr, *CsiAfterRead = nullptr;
  Function *CsiBeforeWrite = nullptr, *CsiAfterWrite = nullptr;
  Function *CsiBeforeAlloca = nullptr, *CsiAfterAlloca = nullptr;
  Function *CsiDetach = nullptr, *CsiDetachContinue = nullptr;
  Function *CsiTaskEntry = nullptr, *CsiTaskExit = nullptr;
  Function *CsiBeforeSync = nullptr, *CsiAfterSync = nullptr;

  Function *MemmoveFn = nullptr, *MemcpyFn = nullptr, *MemsetFn = nullptr;
  Function *InitCallsiteToFunction = nullptr;
  // GlobalVariable *DisableInstrGV;

  // Runtime unit initialization
  Function *RTUnitInit = nullptr;

  Type *IntptrTy;
  DenseMap<StringRef, uint64_t> FuncOffsetMap;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_CSI_H
