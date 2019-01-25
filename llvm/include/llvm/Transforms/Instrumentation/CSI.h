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
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {

class Spindle;
class Task;
class TaskInfo;

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
static const char *const CsiAllocFnBaseIdName = "__csi_unit_allocfn_base_id";
static const char *const CsiFreeBaseIdName = "__csi_unit_free_base_id";

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
static const int CsiUnitCtorPriority = 0;

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

  virtual ~CsiProperty() {}

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

class CsiAllocFnProperty : public CsiProperty {
public:
  CsiAllocFnProperty() {
    PropValue.Bits = 0;
  }
  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.AllocFnTy),
                           IntegerType::get(C, PropBits.Padding)));
  }
  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csan.h
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the allocation function type (e.g., malloc, calloc, new).
  void setAllocFnTy(unsigned v) {
    PropValue.Fields.AllocFnTy = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned AllocFnTy : 8;
      uint64_t Padding : 56;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int AllocFnTy;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 8, (64-8) };
};

class CsiFreeProperty : public CsiProperty {
public:
  CsiFreeProperty() {
    PropValue.Bits = 0;
  }
  /// Return the Type of a property.
  static Type *getType(LLVMContext &C) {
    // Must match the definition of property type in csi.h
    return CsiProperty::getCoercedType(
        C, StructType::get(IntegerType::get(C, PropBits.FreeTy),
                           IntegerType::get(C, PropBits.Padding)));
  }
  /// Return a constant value holding this property.
  Constant *getValueImpl(LLVMContext &C) const override {
    // Must match the definition of property type in csan.h
    return ConstantInt::get(getType(C), PropValue.Bits);
  }

  /// Set the value of the allocation function type (e.g., malloc, calloc, new).
  void setFreeTy(unsigned v) {
    PropValue.Fields.FreeTy = v;
  }

private:
  typedef union {
    // Must match the definition of property type in csi.h
    struct {
      unsigned FreeTy : 8;
      uint64_t Padding : 56;
    } Fields;
    uint64_t Bits;
  } Property;

  /// The underlying values of the properties.
  Property PropValue;

  typedef struct {
    int FreeTy;
    int Padding;
  } PropertyBits;

  /// The number of bits representing each property.
  static constexpr PropertyBits PropBits = { 8, (64-8) };
};

struct CSIImpl {
public:
  CSIImpl(Module &M, CallGraph *CG,
          function_ref<DominatorTree &(Function &)> GetDomTree,
          function_ref<TaskInfo &(Function &)> GetTaskInfo,
          const TargetLibraryInfo *TLI,
          const CSIOptions &Options = CSIOptions())
      : M(M), DL(M.getDataLayout()), CG(CG), GetDomTree(GetDomTree),
        GetTaskInfo(GetTaskInfo), TLI(TLI), Options(Options)
  {}

  virtual ~CSIImpl() {}

  bool run();

  /// Get the number of bytes accessed via the given address.
  static int getNumBytesAccessed(Value *Addr, const DataLayout &DL);

  /// Members to extract properties of loads/stores.
  static bool isVtableAccess(Instruction *I);
  static bool addrPointsToConstantData(Value *Addr);
  static bool isAtomic(Instruction *I);
  static void getAllocFnArgs(
      const Instruction *I, SmallVectorImpl<Value*> &AllocFnArgs,
      Type *SizeTy, Type *AddrTy, const TargetLibraryInfo &TLI);

  /// Helper functions to deal with calls to functions that can throw.
  static void setupCalls(Function &F);
  static void setupBlocks(Function &F, const TargetLibraryInfo *TLI,
                          DominatorTree *DT = nullptr);

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
  void initializeAllocFnHooks();
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
  void instrumentBasicBlock(BasicBlock &BB);
  void instrumentDetach(DetachInst *DI, DominatorTree *DT, TaskInfo &TI);
  void instrumentSync(SyncInst *SI);
  void instrumentAlloca(Instruction *I);
  void instrumentAllocFn(Instruction *I, DominatorTree *DT);
  void instrumentFree(Instruction *I);

  void instrumentFunction(Function &F);
  /// @}

  /// Insert a call to the given hook function before the given instruction.
  void insertHookCall(Instruction *I, Function *HookFunction,
                      ArrayRef<Value *> HookArgs);
  bool updateArgPHIs(
      BasicBlock *Succ, BasicBlock *BB, ArrayRef<Value *> HookArgs,
      ArrayRef<Value *> DefaultHookArgs);
  void insertHookCallInSuccessorBB(
      BasicBlock *Succ, BasicBlock *BB, Function *HookFunction,
      ArrayRef<Value *> HookArgs, ArrayRef<Value *> DefaultHookArgs);
  void insertHookCallAtSharedEHSpindleExits(
      Spindle *SharedEHSpindle, Task *T, Function *HookFunction,
      FrontEndDataTable &FED,
      ArrayRef<Value *> HookArgs, ArrayRef<Value *> DefaultArgs);

  /// Return true if the given function should not be instrumented.
  bool shouldNotInstrumentFunction(Function &F);

  // Update the attributes on the instrumented function that might be
  // invalidated by the inserted instrumentation.
  void updateInstrumentedFnAttrs(Function &F);
  // List of all allocation function types.  This list needs to remain
  // consistent with TargetLibraryInfo and with csan.h.
  enum class AllocFnTy
    {
     malloc = 0,
     valloc,
     calloc,
     realloc,
     reallocf,
     Znwj,
     ZnwjRKSt9nothrow_t,
     Znwm,
     ZnwmRKSt9nothrow_t,
     Znaj,
     ZnajRKSt9nothrow_t,
     Znam,
     ZnamRKSt9nothrow_t,
     msvc_new_int,
     msvc_new_int_nothrow,
     msvc_new_longlong,
     msvc_new_longlong_nothrow,
     msvc_new_array_int,
     msvc_new_array_int_nothrow,
     msvc_new_array_longlong,
     msvc_new_array_longlong_nothrow,
     LAST_ALLOCFNTY
    };

  static AllocFnTy getAllocFnTy(const LibFunc &F) {
    switch (F) {
    default: return AllocFnTy::LAST_ALLOCFNTY;
    case LibFunc_malloc: return AllocFnTy::malloc;
    case LibFunc_valloc: return AllocFnTy::valloc;
    case LibFunc_calloc: return AllocFnTy::calloc;
    case LibFunc_realloc: return AllocFnTy::realloc;
    case LibFunc_reallocf: return AllocFnTy::reallocf;
    case LibFunc_Znwj: return AllocFnTy::Znwj;
    case LibFunc_ZnwjRKSt9nothrow_t: return AllocFnTy::ZnwjRKSt9nothrow_t;
    case LibFunc_Znwm: return AllocFnTy::Znwm;
    case LibFunc_ZnwmRKSt9nothrow_t: return AllocFnTy::ZnwmRKSt9nothrow_t;
    case LibFunc_Znaj: return AllocFnTy::Znaj;
    case LibFunc_ZnajRKSt9nothrow_t: return AllocFnTy::ZnajRKSt9nothrow_t;
    case LibFunc_Znam: return AllocFnTy::Znam;
    case LibFunc_ZnamRKSt9nothrow_t: return AllocFnTy::ZnamRKSt9nothrow_t;
    case LibFunc_msvc_new_int: return AllocFnTy::msvc_new_int;
    case LibFunc_msvc_new_int_nothrow: return AllocFnTy::msvc_new_int_nothrow;
    case LibFunc_msvc_new_longlong: return AllocFnTy::msvc_new_longlong;
    case LibFunc_msvc_new_longlong_nothrow:
      return AllocFnTy::msvc_new_longlong_nothrow;
    case LibFunc_msvc_new_array_int: return AllocFnTy::msvc_new_array_int;
    case LibFunc_msvc_new_array_int_nothrow:
      return AllocFnTy::msvc_new_array_int_nothrow;
    case LibFunc_msvc_new_array_longlong:
      return AllocFnTy::msvc_new_array_longlong;
    case LibFunc_msvc_new_array_longlong_nothrow:
      return AllocFnTy::msvc_new_array_longlong_nothrow;
    }
  }

  // List of all free function types.  This list needs to remain consistent with
  // TargetLibraryInfo and with csi.h.
  enum class FreeTy
    {
     free = 0,
     ZdlPv,
     ZdlPvRKSt9nothrow_t,
     ZdlPvj,
     ZdlPvm,
     ZdaPv,
     ZdaPvRKSt9nothrow_t,
     ZdaPvj,
     ZdaPvm,
     msvc_delete_ptr32,
     msvc_delete_ptr32_nothrow,
     msvc_delete_ptr32_int,
     msvc_delete_ptr64,
     msvc_delete_ptr64_nothrow,
     msvc_delete_ptr64_longlong,
     msvc_delete_array_ptr32,
     msvc_delete_array_ptr32_nothrow,
     msvc_delete_array_ptr32_int,
     msvc_delete_array_ptr64,
     msvc_delete_array_ptr64_nothrow,
     msvc_delete_array_ptr64_longlong,
     LAST_FREETY
    };

  static FreeTy getFreeTy(const LibFunc &F) {
    switch (F) {
    default: return FreeTy::LAST_FREETY;
    case LibFunc_free: return FreeTy::free;
    case LibFunc_ZdlPv: return FreeTy::ZdlPv;
    case LibFunc_ZdlPvRKSt9nothrow_t: return FreeTy::ZdlPvRKSt9nothrow_t;
    case LibFunc_ZdlPvj: return FreeTy::ZdlPvj;
    case LibFunc_ZdlPvm: return FreeTy::ZdlPvm;
    case LibFunc_ZdaPv: return FreeTy::ZdaPv;
    case LibFunc_ZdaPvRKSt9nothrow_t: return FreeTy::ZdaPvRKSt9nothrow_t;
    case LibFunc_ZdaPvj: return FreeTy::ZdaPvj;
    case LibFunc_ZdaPvm: return FreeTy::ZdaPvm;
    case LibFunc_msvc_delete_ptr32: return FreeTy::msvc_delete_ptr32;
    case LibFunc_msvc_delete_ptr32_nothrow:
      return FreeTy::msvc_delete_ptr32_nothrow;
    case LibFunc_msvc_delete_ptr32_int: return FreeTy::msvc_delete_ptr32_int;
    case LibFunc_msvc_delete_ptr64: return FreeTy::msvc_delete_ptr64;
    case LibFunc_msvc_delete_ptr64_nothrow:
      return FreeTy::msvc_delete_ptr64_nothrow;
    case LibFunc_msvc_delete_ptr64_longlong:
      return FreeTy::msvc_delete_ptr64_longlong;
    case LibFunc_msvc_delete_array_ptr32:
      return FreeTy::msvc_delete_array_ptr32;
    case LibFunc_msvc_delete_array_ptr32_nothrow:
      return FreeTy::msvc_delete_array_ptr32_nothrow;
    case LibFunc_msvc_delete_array_ptr32_int:
      return FreeTy::msvc_delete_array_ptr32_int;
    case LibFunc_msvc_delete_array_ptr64:
      return FreeTy::msvc_delete_array_ptr64;
    case LibFunc_msvc_delete_array_ptr64_nothrow:
      return FreeTy::msvc_delete_array_ptr64_nothrow;
    case LibFunc_msvc_delete_array_ptr64_longlong:
      return FreeTy::msvc_delete_array_ptr64_longlong;
    }
  }

  Module &M;
  const DataLayout &DL;
  CallGraph *CG;
  function_ref<DominatorTree &(Function &)> GetDomTree;
  function_ref<TaskInfo &(Function &)> GetTaskInfo;
  const TargetLibraryInfo *TLI;
  CSIOptions Options;

  FrontEndDataTable FunctionFED, FunctionExitFED, BasicBlockFED, CallsiteFED,
    LoadFED, StoreFED, 
    AllocaFED, 
    DetachFED, TaskFED, TaskExitFED, DetachContinueFED,
    SyncFED,
    AllocFnFED, FreeFED;

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
  Function *CsiBeforeAllocFn = nullptr, *CsiAfterAllocFn = nullptr;
  Function *CsiBeforeFree = nullptr, *CsiAfterFree = nullptr;

  Function *MemmoveFn = nullptr, *MemcpyFn = nullptr, *MemsetFn = nullptr;
  Function *InitCallsiteToFunction = nullptr;
  // GlobalVariable *DisableInstrGV;

  // Runtime unit initialization
  Function *RTUnitInit = nullptr;

  Type *IntptrTy;
  DenseMap<StringRef, uint64_t> FuncOffsetMap;

  DenseMap<BasicBlock *, SmallVector<PHINode *, 4>> ArgPHIs;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_CSI_H
