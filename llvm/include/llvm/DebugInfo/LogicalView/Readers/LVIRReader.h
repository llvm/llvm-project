//===-- LVIRReader.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LVIRReader class, which is used to describe a
// LLVM IR reader.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_READERS_LVIRREADER_H
#define LLVM_DEBUGINFO_LOGICALVIEW_READERS_LVIRREADER_H

#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"
#include "llvm/Transforms/Utils/DebugSSAUpdater.h"

namespace llvm {
class DIFile;
class DINode;
class DILocation;
class DIScope;
class DISubprogram;
class DIVariable;
class BasicBlock;
class Module;

namespace object {
class IRObjectFile;
}

namespace logicalview {

class LVElement;
class LVLine;
class LVScopeCompileUnit;
class LVSymbol;
class LVType;
struct LVSourceLanguage;

class LVIRReader final : public LVReader {
  object::IRObjectFile *BitCodeIR = nullptr;
  MemoryBufferRef *TextualIR = nullptr;

  // Used by the metadata 'dump' functions, so the metadata nodes will be
  // numbered canonically; otherwise, pointer addresses are substituted.
  Module *TheModule = nullptr;

  // Symbols with locations for current compile unit.
  LVSymbols SymbolsWithLocations;

  const DICompileUnit *CUNode = nullptr;

  // The Dwarf version (from the module flags).
  unsigned DwarfVersion = 0;

  // Location index for global variables.
  uint64_t PoolAddressIndex = 0;

  // Whether to emit all linkage names, or just abstract subprograms.
  bool UseAllLinkageNames = true;

  // Looking at IR generated with the '-gdwarf -gsplit-dwarf=split' the only
  // difference is setting the 'DICompileUnit::splitDebugFilename' to the
  // name of the split filename: "xxx.dwo".
  bool includeMinimalInlineScopes() const;
  bool useAllLinkageNames() const { return UseAllLinkageNames; }

  // Default lower bound for arrays.
  int64_t DefaultLowerBound = 0;
  int64_t getDefaultLowerBound() const { return DefaultLowerBound; }
  void setDefaultLowerBound(LVSourceLanguage *SL);

  // We assume a constant instruction-size increase between instructions.
  const unsigned OffsetIncrease = 4;
  void updateLineOffset() { CurrentOffset += OffsetIncrease; }

  SSAValueNameMap ValueNameMap;
  DenseMap<void *, uint64_t> InstrLineAddrMap;
  std::unique_ptr<DbgValueRangeTable> DbgValueRanges;

  // Record the last assigned file index for each compile unit.
  // This data structure is to aid mapping DIFiles onto a DWARF-like file table.
  using LVIndexFiles = std::map<LVScopeCompileUnit *, size_t>;
  LVIndexFiles IndexFiles;

  void updateFileIndex(LVScopeCompileUnit *CompileUnit, size_t FileIndex) {
    LVIndexFiles::iterator Iter = IndexFiles.find(CompileUnit);
    if (Iter == IndexFiles.end())
      IndexFiles.emplace(CompileUnit, FileIndex);
    else
      Iter->second = FileIndex;
  }

  // Get the current assigned index file for the given compile unit.
  size_t getFileIndex(LVScopeCompileUnit *CompileUnit) {
    size_t FileIndex = 0;
    LVIndexFiles::iterator Iter = IndexFiles.find(CompileUnit);
    if (Iter != IndexFiles.end())
      FileIndex = Iter->second;
    return FileIndex;
  }

  // Store a FileID number for each DIFile seen.
  using LVCompileUnitFiles = std::map<const DIFile *, size_t>;
  LVCompileUnitFiles CompileUnitFiles;

  // For the given 'DIFile', generate a 1-based index to indicate the
  // source file where the logical element is declared.
  // The IR reader expects the indexes to be 1-based.
  // Each compile unit, keeps track of the last assigned index.
  size_t getOrCreateSourceID(const DIFile *File);

  // Associate the metadata objects to logical elements.
  using LVMDObjects = std::map<const MDNode *, LVElement *>;
  LVMDObjects MDObjects;

  void addMD(const MDNode *MD, LVElement *Element) {
    MDObjects.try_emplace(MD, Element);
  }
  LVElement *getElementForSeenMD(const MDNode *MD) const {
    LVMDObjects::const_iterator Iter = MDObjects.find(MD);
    return Iter != MDObjects.end() ? Iter->second : nullptr;
  }
  LVScope *getScopeForSeenMD(const MDNode *MD) const {
    return static_cast<LVScope *>(getElementForSeenMD(MD));
  }
  LVSymbol *getSymbolForSeenMD(const MDNode *MD) const {
    return static_cast<LVSymbol *>(getElementForSeenMD(MD));
  }
  LVType *getTypeForSeenMD(const MDNode *MD) const {
    return static_cast<LVType *>(getElementForSeenMD(MD));
  }
  LVType *getLineForSeenMD(const MDNode *MD) const {
    return static_cast<LVType *>(getElementForSeenMD(MD));
  }

  const DIFile *getMDFile(const MDNode *MD) const;
  StringRef getMDName(const DINode *DN) const;
  const DIScope *getMDScope(const DINode *DN) const;

  // An anonymous type for index type.
  LVType *NodeIndexType = nullptr;
  LVType *getIndexType();

  // Get current metadata DICompileUnit object.
  const DICompileUnit *getCUNode() const { return CUNode; }

  // Get the parent scope for the given metadata object.
  LVScope *getParentScopeImpl(const DIScope *Context);
  LVScope *getParentScope(const DINode *DN);
  LVScope *getParentScope(const DILocation *DL);

  // Traverse the scope hierarchy and create each node in the hierarchy.
  LVScope *traverseParentScope(const DIScope *Context);

  // Create the location ranges for the given scope and in the case of
  // functions, generate an entry in the public names set.
  void constructRange(LVScope *Scope, LVAddress LowPC, LVAddress HighPC);
  void constructRange(LVScope *Scope);

  // Generate debug logical lines for the given function.
  void processBasicBlocks(Function &F);

  // Add accessibility information if available.
  void addAccess(LVElement *Element, DINode::DIFlags Flags);

  // Add a constant value to a logical element.
  void addConstantValue(LVElement *Element, const DIExpression *DIExpr);
  void addConstantValue(LVElement *Element, const ConstantInt *CI,
                        const DIType *Ty);
  void addConstantValue(LVElement *Element, const APInt &Value,
                        const DIType *Ty);
  void addConstantValue(LVElement *Element, const APInt &Value, bool Unsigned);
  void addConstantValue(LVElement *Element, uint64_t Value, const DIType *Ty);
  void addConstantValue(LVElement *Element, uint64_t Value, bool Unsigned);

  // Add template parameters to logical element.
  void addTemplateParams(LVElement *Element, const DINodeArray TParams);

  // Add location information to specified logical element.
  void addSourceLine(LVElement *Element, unsigned Line, const DIFile *File);
  void addSourceLine(LVElement *Element, const DIGlobalVariable *GV);
  void addSourceLine(LVElement *Element, const DIImportedEntity *IE);
  void addSourceLine(LVElement *Element, const DILabel *L);
  void addSourceLine(LVElement *Element, const DILocalVariable *LV);
  void addSourceLine(LVElement *Element, const DILocation *DL);
  void addSourceLine(LVElement *Element, const DIObjCProperty *OP);
  void addSourceLine(LVElement *Element, const DISubprogram *SP);
  void addSourceLine(LVElement *Element, const DIType *Ty);

  void applySubprogramAttributes(LVScope *Function, const DISubprogram *SP,
                                 bool SkipSPAttributes = false);
  bool applySubprogramDefinitionAttributes(LVScope *Function,
                                           const DISubprogram *SP,
                                           bool Minimal = false);

  void constructAggregate(LVScopeAggregate *Aggregate,
                          const DICompositeType *CTy);
  void constructArray(LVScopeArray *Array, const DICompositeType *CTy);
  void constructEnum(LVScopeEnumeration *Enumeration,
                     const DICompositeType *CTy);
  void constructGenericSubrange(LVScopeArray *Array,
                                const DIGenericSubrange *GSR,
                                LVType *IndexType);
  void constructImportedEntity(LVElement *Element, const DIImportedEntity *IE);

  void constructLine(LVScope *Scope, const DISubprogram *SP, Instruction &I,
                     bool &GenerateLineBeforePrologue);

  LVSymbol *getOrCreateMember(LVScope *Aggregate, const DIDerivedType *DT);
  LVSymbol *getOrCreateStaticMember(LVScope *Aggregate,
                                    const DIDerivedType *DT);

  LVScope *getOrCreateNamespace(const DINamespace *NS);
  LVScope *getOrCreateScope(const DIScope *Context);
  void constructScope(LVElement *Element, const DIScope *Context);

  LVScope *getOrCreateSubprogram(const DISubprogram *SP);
  LVScope *getOrCreateSubprogram(LVScope *Function, const DISubprogram *SP,
                                 bool Minimal = false);
  void constructSubprogramArguments(LVScope *Function, const DITypeArray Args);

  LVScope *getOrCreateAbstractScope(const DILocation *DL);
  LVScope *getOrCreateInlinedScope(const DILocation *DL);

  void constructSubrange(LVScopeArray *Array, const DISubrange *SR,
                         LVType *IndexType);

  void constructTemplateTypeParameter(LVElement *Element,
                                      const DITemplateTypeParameter *TTP);
  void constructTemplateValueParameter(LVElement *Element,
                                       const DITemplateValueParameter *TVP);

  LVElement *getOrCreateType(const DIType *Ty, LVScope *Scope = nullptr);
  void constructType(LVScope *Scope, const DICompositeType *CTy);
  void constructType(LVElement *Element, const DIDerivedType *DT);
  void constructType(LVScope *Function, const DISubroutineType *SPTy);

  LVSymbol *getOrCreateVariable(const DIGlobalVariableExpression *GVE);
  LVSymbol *getOrCreateVariable(const DIVariable *Var,
                                const DILocation *DL = nullptr);
  LVSymbol *getOrCreateInlinedVariable(LVSymbol *OriginSymbol,
                                       const DILocation *DL);

  LVElement *constructElement(const DINode *DN);

  // After the scopes have been created, remove empty ones.
  void removeEmptyScopes();

  // Adjust the inlined lexical scopes to their correct scope.
  void resolveInlinedLexicalScopes();

  void processLocationGaps();
  void processScopes();

  // Check if the scopes have been properly constructed (finalized).
  void checkScopes(LVScope *Scope);

protected:
  Error createScopes() override;
  void sortScopes() override;

public:
  LVIRReader() = delete;
  LVIRReader(StringRef Filename, StringRef FileFormatName,
             object::IRObjectFile *Obj, ScopedPrinter &W)
      : LVReader(Filename, FileFormatName, W, LVBinaryType::ELF),
        BitCodeIR(Obj), DbgValueRanges(new DbgValueRangeTable()) {}
  LVIRReader(StringRef Filename, StringRef FileFormatName, MemoryBufferRef *Obj,
             ScopedPrinter &W)
      : LVReader(Filename, FileFormatName, W, LVBinaryType::ELF),
        TextualIR(Obj), DbgValueRanges(new DbgValueRangeTable()) {}
  LVIRReader(const LVIRReader &) = delete;
  LVIRReader &operator=(const LVIRReader &) = delete;
  ~LVIRReader() = default;

  const LVSymbols &GetSymbolsWithLocations() const {
    return SymbolsWithLocations;
  }

  std::string getRegisterName(LVSmall Opcode,
                              ArrayRef<uint64_t> Operands) override;

  void print(raw_ostream &OS) const;
#ifdef LLVM_DEBUG
  void printAllInstructions(BasicBlock *BB);
#else
#define printAllInstructions(...) (void *)0
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const { print(dbgs()); }
#endif
};

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_READERS_LVIRREADER_H
