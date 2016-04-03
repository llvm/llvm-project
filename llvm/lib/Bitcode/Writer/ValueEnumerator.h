//===-- Bitcode/Writer/ValueEnumerator.h - Number values --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class gives values and types Unique ID's.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_BITCODE_WRITER_VALUEENUMERATOR_H
#define LLVM_LIB_BITCODE_WRITER_VALUEENUMERATOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/UseListOrder.h"
#include <vector>

namespace llvm {

class Type;
class Value;
class Instruction;
class BasicBlock;
class Comdat;
class Function;
class Module;
class Metadata;
class LocalAsMetadata;
class MDNode;
class NamedMDNode;
class AttributeSet;
class ValueSymbolTable;
class MDSymbolTable;
class raw_ostream;

class ValueEnumerator {
public:
  typedef std::vector<Type*> TypeList;

  // For each value, we remember its Value* and occurrence frequency.
  typedef std::vector<std::pair<const Value*, unsigned> > ValueList;

  UseListOrderStack UseListOrders;

private:
  typedef DenseMap<Type*, unsigned> TypeMapType;
  TypeMapType TypeMap;
  TypeList Types;

  typedef DenseMap<const Value*, unsigned> ValueMapType;
  ValueMapType ValueMap;
  ValueList Values;

  typedef UniqueVector<const Comdat *> ComdatSetType;
  ComdatSetType Comdats;

  std::vector<const Metadata *> MDs;
  std::vector<const Metadata *> FunctionMDs;

  /// Index of information about a piece of metadata.
  struct MDIndex {
    unsigned F = 0;  ///< The ID of the function for this metadata, if any.
    unsigned ID = 0; ///< The implicit ID of this metadata in bitcode.

    MDIndex() = default;
    explicit MDIndex(unsigned F) : F(F) {}

    /// Check if this has a function tag, and it's different from NewF.
    bool hasDifferentFunction(unsigned NewF) const { return F && F != NewF; }

    /// Fetch the MD this references out of the given metadata array.
    const Metadata *get(ArrayRef<const Metadata *> MDs) const {
      assert(ID && "Expected non-zero ID");
      assert(ID <= MDs.size() && "Expected valid ID");
      return MDs[ID - 1];
    }
  };
  typedef DenseMap<const Metadata *, MDIndex> MetadataMapType;
  MetadataMapType MetadataMap;

  /// Range of metadata IDs, as a half-open range.
  struct MDRange {
    unsigned First = 0;
    unsigned Last = 0;

    /// Number of strings in the prefix of the metadata range.
    unsigned NumStrings = 0;

    MDRange() = default;
    explicit MDRange(unsigned First) : First(First) {}
  };
  SmallDenseMap<unsigned, MDRange, 1> FunctionMDInfo;

  bool ShouldPreserveUseListOrder;

  typedef DenseMap<AttributeSet, unsigned> AttributeGroupMapType;
  AttributeGroupMapType AttributeGroupMap;
  std::vector<AttributeSet> AttributeGroups;

  typedef DenseMap<AttributeSet, unsigned> AttributeMapType;
  AttributeMapType AttributeMap;
  std::vector<AttributeSet> Attribute;

  /// GlobalBasicBlockIDs - This map memoizes the basic block ID's referenced by
  /// the "getGlobalBasicBlockID" method.
  mutable DenseMap<const BasicBlock*, unsigned> GlobalBasicBlockIDs;

  typedef DenseMap<const Instruction*, unsigned> InstructionMapType;
  InstructionMapType InstructionMap;
  unsigned InstructionCount;

  /// BasicBlocks - This contains all the basic blocks for the currently
  /// incorporated function.  Their reverse mapping is stored in ValueMap.
  std::vector<const BasicBlock*> BasicBlocks;

  /// When a function is incorporated, this is the size of the Values list
  /// before incorporation.
  unsigned NumModuleValues;

  /// When a function is incorporated, this is the size of the Metadatas list
  /// before incorporation.
  unsigned NumModuleMDs = 0;
  unsigned NumMDStrings = 0;

  unsigned FirstFuncConstantID;
  unsigned FirstInstID;

  ValueEnumerator(const ValueEnumerator &) = delete;
  void operator=(const ValueEnumerator &) = delete;
public:
  ValueEnumerator(const Module &M, bool ShouldPreserveUseListOrder);

  void dump() const;
  void print(raw_ostream &OS, const ValueMapType &Map, const char *Name) const;
  void print(raw_ostream &OS, const MetadataMapType &Map,
             const char *Name) const;

  unsigned getValueID(const Value *V) const;
  unsigned getMetadataID(const Metadata *MD) const {
    auto ID = getMetadataOrNullID(MD);
    assert(ID != 0 && "Metadata not in slotcalculator!");
    return ID - 1;
  }
  unsigned getMetadataOrNullID(const Metadata *MD) const {
    return MetadataMap.lookup(MD).ID;
  }
  unsigned numMDs() const { return MDs.size(); }

  bool shouldPreserveUseListOrder() const { return ShouldPreserveUseListOrder; }

  unsigned getTypeID(Type *T) const {
    TypeMapType::const_iterator I = TypeMap.find(T);
    assert(I != TypeMap.end() && "Type not in ValueEnumerator!");
    return I->second-1;
  }

  unsigned getInstructionID(const Instruction *I) const;
  void setInstructionID(const Instruction *I);

  unsigned getAttributeID(AttributeSet PAL) const {
    if (PAL.isEmpty()) return 0;  // Null maps to zero.
    AttributeMapType::const_iterator I = AttributeMap.find(PAL);
    assert(I != AttributeMap.end() && "Attribute not in ValueEnumerator!");
    return I->second;
  }

  unsigned getAttributeGroupID(AttributeSet PAL) const {
    if (PAL.isEmpty()) return 0;  // Null maps to zero.
    AttributeGroupMapType::const_iterator I = AttributeGroupMap.find(PAL);
    assert(I != AttributeGroupMap.end() && "Attribute not in ValueEnumerator!");
    return I->second;
  }

  /// getFunctionConstantRange - Return the range of values that corresponds to
  /// function-local constants.
  void getFunctionConstantRange(unsigned &Start, unsigned &End) const {
    Start = FirstFuncConstantID;
    End = FirstInstID;
  }

  const ValueList &getValues() const { return Values; }

  /// Check whether the current block has any metadata to emit.
  bool hasMDs() const { return NumModuleMDs < MDs.size(); }

  /// Get the MDString metadata for this block.
  ArrayRef<const Metadata *> getMDStrings() const {
    return makeArrayRef(MDs).slice(NumModuleMDs, NumMDStrings);
  }

  /// Get the non-MDString metadata for this block.
  ArrayRef<const Metadata *> getNonMDStrings() const {
    return makeArrayRef(MDs).slice(NumModuleMDs).slice(NumMDStrings);
  }

  const TypeList &getTypes() const { return Types; }
  const std::vector<const BasicBlock*> &getBasicBlocks() const {
    return BasicBlocks;
  }
  const std::vector<AttributeSet> &getAttributes() const {
    return Attribute;
  }
  const std::vector<AttributeSet> &getAttributeGroups() const {
    return AttributeGroups;
  }

  const ComdatSetType &getComdats() const { return Comdats; }
  unsigned getComdatID(const Comdat *C) const;

  /// getGlobalBasicBlockID - This returns the function-specific ID for the
  /// specified basic block.  This is relatively expensive information, so it
  /// should only be used by rare constructs such as address-of-label.
  unsigned getGlobalBasicBlockID(const BasicBlock *BB) const;

  /// incorporateFunction/purgeFunction - If you'd like to deal with a function,
  /// use these two methods to get its data into the ValueEnumerator!
  ///
  void incorporateFunction(const Function &F);
  void purgeFunction();
  uint64_t computeBitsRequiredForTypeIndicies() const;

private:
  void OptimizeConstants(unsigned CstStart, unsigned CstEnd);

  /// Reorder the reachable metadata.
  ///
  /// This is not just an optimization, but is mandatory for emitting MDString
  /// correctly.
  void organizeMetadata();

  /// Drop the function tag from the transitive operands of the given node.
  void dropFunctionFromOps(const MDNode &N);

  /// Incorporate the function metadata.
  ///
  /// This should be called before enumerating LocalAsMetadata for the
  /// function.
  void incorporateFunctionMetadata(const Function &F);

  bool insertMetadata(unsigned F, const Metadata *MD);

  unsigned getMetadataFunctionID(const Function *F) const;
  void EnumerateMDNodeOperands(const Function *F, const MDNode *N);
  void EnumerateMDNodeOperands(unsigned F, const MDNode *N);
  void EnumerateMetadata(const Function *F, const Metadata *MD);
  void EnumerateMetadata(unsigned F, const Metadata *MD);
  void EnumerateFunctionLocalMetadata(const Function &F,
                                      const LocalAsMetadata *Local);
  void EnumerateFunctionLocalMetadata(unsigned F, const LocalAsMetadata *Local);
  void EnumerateNamedMDNode(const NamedMDNode *NMD);
  void EnumerateValue(const Value *V);
  void EnumerateType(Type *T);
  void EnumerateOperandType(const Value *V);
  void EnumerateAttributes(AttributeSet PAL);

  void EnumerateValueSymbolTable(const ValueSymbolTable &ST);
  void EnumerateNamedMetadata(const Module &M);
};

} // End llvm namespace

#endif
