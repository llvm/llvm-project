//===- BTFDebug.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains support for writing BTF debug info.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BTFDEBUG_H
#define LLVM_LIB_TARGET_BPF_BTFDEBUG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/DebugHandlerBase.h"
#include "llvm/DebugInfo/BTF/BTF.h"
#include <cstdint>
#include <map>
#include <optional>
#include <set>

namespace llvm {

class AsmPrinter;
class BTFDebug;
class DIType;
class GlobalVariable;
class MachineFunction;
class MachineInstr;
class MachineOperand;
class MCInst;
class MCStreamer;
class MCSymbol;

/// The base class for BTF type generation.
class BTFTypeBase {
protected:
  bool IsCompleted = false;
  uint32_t Id;
  BTF::CommonType BTFType;

  explicit BTFTypeBase(BTF::TypeKinds Kind, uint32_t VLen = 0,
                       bool KindFlag = false);

public:
  virtual ~BTFTypeBase() = default;
  void setId(uint32_t Id) { this->Id = Id; }
  uint32_t getId() { return Id; }
  uint32_t roundupToBytes(uint32_t NumBits) { return (NumBits + 7) >> 3; }
  /// Get the size of this BTF type entry.
  virtual uint32_t getSize() { return BTF::CommonTypeSize; }
  /// Complete BTF type generation after all related DebugInfo types
  /// have been visited so their BTF type id's are available
  /// for cross referece.
  virtual void completeType(BTFDebug &BDebug) {}
  /// Emit types for this BTF type entry.
  virtual void emitType(MCStreamer &OS);
};

/// Handle several derived types include pointer, const,
/// volatile, typedef and restrict.
class BTFTypeDerived : public BTFTypeBase {
  const DIDerivedType *DTy;
  bool NeedsFixup;
  StringRef Name;

public:
  BTFTypeDerived(const DIDerivedType *Ty, BTF::TypeKinds Kind, bool NeedsFixup);
  BTFTypeDerived(uint32_t NextTypeId, BTF::TypeKinds Kind, StringRef Name);
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
  void setPointeeType(uint32_t PointeeType);
};

/// Handle struct or union forward declaration.
class BTFTypeFwd : public BTFTypeBase {
  StringRef Name;

public:
  BTFTypeFwd(StringRef Name, bool IsUnion);
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle int type.
class BTFTypeInt : public BTFTypeBase {
  StringRef Name;
  uint32_t IntVal; ///< Encoding, offset, bits

public:
  BTFTypeInt(uint32_t Encoding, uint32_t SizeInBits, uint32_t OffsetInBits,
             StringRef TypeName);
  uint32_t getSize() override { return BTFTypeBase::getSize() + sizeof(uint32_t); }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle enumerate type.
class BTFTypeEnum : public BTFTypeBase {
  const DICompositeType *ETy;
  std::vector<struct BTF::BTFEnum> EnumValues;

public:
  BTFTypeEnum(const DICompositeType *ETy, uint32_t NumValues, bool IsSigned);
  uint32_t getSize() override {
    return BTFTypeBase::getSize() + EnumValues.size() * BTF::BTFEnumSize;
  }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle array type.
class BTFTypeArray : public BTFTypeBase {
  struct BTF::BTFArray ArrayInfo;

public:
  BTFTypeArray(uint32_t ElemTypeId, uint32_t NumElems);
  uint32_t getSize() override { return BTFTypeBase::getSize() + BTF::BTFArraySize; }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle struct/union type.
class BTFTypeStruct : public BTFTypeBase {
  const DICompositeType *STy;
  std::vector<const DINode *> Elements;
  bool HasBitField;
  std::vector<struct BTF::BTFMember> Members;

public:
  BTFTypeStruct(const DICompositeType *STy, ArrayRef<const DINode *> Elements,
                bool IsStruct, bool HasBitField, uint32_t NumMembers);
  uint32_t getSize() override {
    return BTFTypeBase::getSize() + Members.size() * BTF::BTFMemberSize;
  }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
  std::string getName();
};

/// Handle function pointer.
class BTFTypeFuncProto : public BTFTypeBase {
  const DISubroutineType *STy;
  SmallDenseMap<uint32_t, StringRef> FuncArgNames;
  SmallVector<uint32_t, 8> AliveParamIndices;
  bool UseFilteredParams = false;
  std::vector<struct BTF::BTFParam> Parameters;
  bool VoidReturn = false;

public:
  BTFTypeFuncProto(const DISubroutineType *STy, uint32_t NumParams,
                   const SmallDenseMap<uint32_t, StringRef> &FuncArgNames,
                   bool UseFilteredParams = false,
                   ArrayRef<uint32_t> AliveParamIndices = {},
                   bool VoidReturn = false);
  uint32_t getSize() override {
    return BTFTypeBase::getSize() + Parameters.size() * BTF::BTFParamSize;
  }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle subprogram
class BTFTypeFunc : public BTFTypeBase {
  StringRef Name;

public:
  BTFTypeFunc(StringRef FuncName, uint32_t ProtoTypeId, uint32_t Scope);
  uint32_t getSize() override { return BTFTypeBase::getSize(); }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle variable instances
class BTFKindVar : public BTFTypeBase {
  StringRef Name;
  uint32_t Info;

public:
  BTFKindVar(StringRef VarName, uint32_t TypeId, uint32_t VarInfo);
  uint32_t getSize() override { return BTFTypeBase::getSize() + 4; }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle data sections
class BTFKindDataSec : public BTFTypeBase {
  AsmPrinter *Asm;
  std::string Name;
  std::vector<std::tuple<uint32_t, const MCSymbol *, uint32_t>> Vars;

public:
  BTFKindDataSec(AsmPrinter *AsmPrt, std::string SecName);
  uint32_t getSize() override {
    return BTFTypeBase::getSize() + BTF::BTFDataSecVarSize * Vars.size();
  }
  void addDataSecEntry(uint32_t Id, const MCSymbol *Sym, uint32_t Size) {
    Vars.push_back(std::make_tuple(Id, Sym, Size));
  }
  std::string getName() { return Name; }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle binary floating point type.
class BTFTypeFloat : public BTFTypeBase {
  StringRef Name;

public:
  BTFTypeFloat(uint32_t SizeInBits, StringRef TypeName);
  void completeType(BTFDebug &BDebug) override;
};

/// Handle decl tags.
class BTFTypeDeclTag : public BTFTypeBase {
  uint32_t Info;
  StringRef Tag;

public:
  BTFTypeDeclTag(uint32_t BaseTypeId, int ComponentId, StringRef Tag);
  uint32_t getSize() override { return BTFTypeBase::getSize() + 4; }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

/// Handle 64-bit enumerate type.
class BTFTypeEnum64 : public BTFTypeBase {
  const DICompositeType *ETy;
  std::vector<struct BTF::BTFEnum64> EnumValues;

public:
  BTFTypeEnum64(const DICompositeType *ETy, uint32_t NumValues, bool IsSigned);
  uint32_t getSize() override {
    return BTFTypeBase::getSize() + EnumValues.size() * BTF::BTFEnum64Size;
  }
  void completeType(BTFDebug &BDebug) override;
  void emitType(MCStreamer &OS) override;
};

class BTFTypeTypeTag : public BTFTypeBase {
  const DIDerivedType *DTy;
  StringRef Tag;

public:
  BTFTypeTypeTag(uint32_t NextTypeId, StringRef Tag);
  BTFTypeTypeTag(const DIDerivedType *DTy, StringRef Tag);
  void completeType(BTFDebug &BDebug) override;
};

/// String table.
class BTFStringTable {
  /// String table size in bytes.
  uint32_t Size;
  /// A mapping from string table offset to the index
  /// of the Table. It is used to avoid putting
  /// duplicated strings in the table.
  std::map<uint32_t, uint32_t> OffsetToIdMap;
  /// A vector of strings to represent the string table.
  std::vector<std::string> Table;

public:
  BTFStringTable() : Size(0) {}
  uint32_t getSize() { return Size; }
  std::vector<std::string> &getTable() { return Table; }
  /// Add a string to the string table and returns its offset
  /// in the table.
  uint32_t addString(StringRef S);
};

/// Represent one func and its type id.
struct BTFFuncInfo {
  const MCSymbol *Label; ///< Func MCSymbol
  uint32_t TypeId;       ///< Type id referring to .BTF type section
};

/// Represent one line info.
struct BTFLineInfo {
  MCSymbol *Label;      ///< MCSymbol identifying insn for the lineinfo
  uint32_t FileNameOff; ///< file name offset in the .BTF string table
  uint32_t LineOff;     ///< line offset in the .BTF string table
  uint32_t LineNum;     ///< the line number
  uint32_t ColumnNum;   ///< the column number
};

/// Represent one field relocation.
struct BTFFieldReloc {
  const MCSymbol *Label;  ///< MCSymbol identifying insn for the reloc
  uint32_t TypeID;        ///< Type ID
  uint32_t OffsetNameOff; ///< The string to traverse types
  uint32_t RelocKind;     ///< What to patch the instruction
};

/// Collect and emit BTF information.
class BTFDebug : public DebugHandlerBase {
  MCStreamer &OS;
  bool SkipInstruction;
  bool LineInfoGenerated;
  uint32_t SecNameOff;
  uint32_t ArrayIndexTypeId;
  bool MapDefNotCollected;
  BTFStringTable StringTable;
  std::vector<std::unique_ptr<BTFTypeBase>> TypeEntries;
  DenseMap<const DIType *, uint32_t> DIToIdMap;
  bool HasIncompleteFuncInfo = false;
  std::map<uint32_t, std::vector<BTFFuncInfo>> FuncInfoTable;
  std::map<uint32_t, std::vector<BTFLineInfo>> LineInfoTable;
  std::map<uint32_t, std::vector<BTFFieldReloc>> FieldRelocTable;
  StringMap<std::vector<std::string>> FileContent;
  std::map<std::string, std::unique_ptr<BTFKindDataSec>, std::less<>>
      DataSecEntries;
  std::vector<BTFTypeStruct *> StructTypes;
  std::map<const GlobalVariable *, std::pair<int64_t, uint32_t>> PatchImms;
  std::map<const DICompositeType *,
           std::vector<std::pair<const DIDerivedType *, BTFTypeDerived *>>>
      FixupDerivedTypes;
  std::set<const Function *>ProtoFunctions;

  /// Add a BTF record and associate its ID with Ty if supplied.
  uint32_t addType(std::unique_ptr<BTFTypeBase> TypeEntry,
                   const DIType *Ty = nullptr);

  // While traversing a struct or union member, crossing an unannotated pointer
  // makes named record pointees eligible for forward declarations.
  enum class TypeUse { Direct, Member, MemberPointee };

  /// IR type visiting functions.
  /// @{
  std::optional<uint32_t> visitTypeEntry(const DIType *Ty,
                                         TypeUse Use = TypeUse::Direct);
  std::optional<uint32_t> visitBasicType(const DIBasicType *BTy);
  std::optional<uint32_t>
  visitSubroutineType(const DISubroutineType *STy, bool ForSubprog,
                      const SmallDenseMap<uint32_t, StringRef> &FuncArgNames,
                      bool VoidReturn = false);
  uint32_t visitFwdDeclType(const DICompositeType *CTy, bool IsUnion);
  std::optional<uint32_t> visitCompositeType(const DICompositeType *CTy);
  std::optional<uint32_t> visitStructType(const DICompositeType *STy,
                                          bool IsStruct);
  std::optional<uint32_t> visitArrayType(const DICompositeType *ATy);
  std::optional<uint32_t> visitEnumType(const DICompositeType *ETy);
  std::optional<uint32_t> visitDerivedType(const DIDerivedType *DTy,
                                           TypeUse Use);
  std::optional<uint32_t> visitMapDefType(const DIType *Ty);
  /// @}

  /// Check whether the type is a forward declaration candidate or not.
  bool IsForwardDeclCandidate(const DIType *Base);

  /// Get the file content for the subprogram. Certain lines of the file
  /// later may be put into string table and referenced by line info.
  std::string populateFileContent(const DIFile *File);

  /// Construct a line info.
  void constructLineInfo(MCSymbol *Label, const DIFile *File, uint32_t Line,
                         uint32_t Column);

  /// Generate types and variables for globals.
  void processGlobals(bool ProcessingMapDef);

  /// Process global variable initializer in pursuit for function
  /// pointers.
  void processGlobalInitializer(const Constant *C);

  /// Generate types for function prototypes.
  void processFuncPrototypes(const Function *);

  /// Generate types for decl annotations.
  void processDeclAnnotations(DINodeArray Annotations, uint32_t BaseTypeId,
                              int ComponentId);

  /// Generate types for DISubprogram and it's arguments.
  uint32_t processDISubprogram(
      const DISubprogram *SP, uint32_t ProtoTypeId, uint8_t Scope,
      const SmallDenseMap<uint32_t, uint32_t> *ArgIndexMap = nullptr);

  /// Return the head of the BTF type tag chain, if any. The last tag refers
  /// to BaseTypeId when supplied, or to the base type of DTy.
  std::optional<uint32_t>
  genBTFTypeTags(const DIDerivedType *DTy,
                 std::optional<uint32_t> BaseTypeId = std::nullopt);

  /// Generate one field relocation record.
  void generatePatchImmReloc(const MCSymbol *ORSym, uint32_t RootId,
                             const GlobalVariable *, bool IsAma);

  /// Process global variables referenced by relocation instructions
  /// and extern function references.
  void processGlobalValue(const MachineOperand &MO);

  /// Emit common header of .BTF and .BTF.ext sections.
  void emitCommonHeader();

  /// Emit the .BTF section.
  void emitBTFSection();

  /// Emit the .BTF.ext section.
  void emitBTFExtSection();

protected:
  /// Gather pre-function debug information.
  void beginFunctionImpl(const MachineFunction *MF) override;

  /// Post process after all instructions in this function are processed.
  void endFunctionImpl(const MachineFunction *MF) override;

public:
  BTFDebug(AsmPrinter *AP);

  ///
  bool InstLower(const MachineInstr *MI, MCInst &OutMI);

  /// Get the special array index type id.
  uint32_t getArrayIndexTypeId() {
    assert(ArrayIndexTypeId);
    return ArrayIndexTypeId;
  }

  /// Add string to the string table.
  size_t addString(StringRef S) { return StringTable.addString(S); }

  /// Get the ID of a required type referenced by another BTF record.
  uint32_t getTypeId(const DIType *Ty) const;
  uint32_t getDerivedBaseTypeId(const DIDerivedType *DTy) const;

  /// Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI) override;

  /// Complete all the types and emit the BTF sections.
  void endModule() override;
};

} // end namespace llvm

#endif
