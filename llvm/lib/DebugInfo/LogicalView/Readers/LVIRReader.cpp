//===-- LVIRReader.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the LVIRReader class.
// It supports LLVM textual and bitcode IR format.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Readers/LVIRReader.h"
#include "llvm/CodeGen/DebugHandlerBase.h"
#include "llvm/DebugInfo/LogicalView/Core/LVLine.h"
#include "llvm/DebugInfo/LogicalView/Core/LVScope.h"
#include "llvm/DebugInfo/LogicalView/Core/LVSymbol.h"
#include "llvm/DebugInfo/LogicalView/Core/LVType.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::logicalview;

#define DEBUG_TYPE "IRReader"

namespace {

// Abstract scopes mapped to the associated inlined scopes.
// When creating inlined scopes, there is no direct information to find
// the correct lexical scope.
using LVScopeEntry = std::pair<const DILocalScope *, const DILocation *>;
using LVInlinedScopes = std::map<LVScopeEntry, LVScope *>;
LVInlinedScopes InlinedScopes;

void addInlinedScope(const DILocalScope *OriginContext,
                     const DILocation *InlinedAt, LVScope *InlinedScope) {
  auto Entry = LVScopeEntry(OriginContext, InlinedAt);
  InlinedScopes.try_emplace(Entry, InlinedScope);
}
LVScope *getInlinedScope(const DILocalScope *OriginContext,
                         const DILocation *InlinedAt) {
  auto Entry = LVScopeEntry(OriginContext, InlinedAt);
  LVInlinedScopes::const_iterator Iter = InlinedScopes.find(Entry);
  return Iter != InlinedScopes.end() ? Iter->second : nullptr;
}

// Used to find the correct location for the inlined lexical blocks that
// are allocated at their enclosing function level.
// Keep a link between the inlined scope and its associated origin scope.
using LVInlinedToOrigin = std::map<LVScope *, LVScope *>;
LVInlinedToOrigin InlinedToOrigin;

// Keep a list of inlined scopes created from the same origin scope.
// The original scope can be inlined multiple times.
using LVList = llvm::SmallVector<LVScope *, 2>;
using LVInlinedList = std::map<LVScope *, LVList>;
LVInlinedList InlinedList;

void addInlinedInfo(LVScope *Origin, LVScope *Inlined) {
  // Add the link between the inlined and the origin scopes.
  InlinedToOrigin.try_emplace(Inlined, Origin);

  // For the given origin scope, add the inlined scope to its inlined list.
  auto It = InlinedList.find(Origin);
  if (It == InlinedList.end()) {
    LVList List;
    List.push_back(Inlined);
    InlinedList.try_emplace(Origin, std::move(List));
  } else {
    LVList &List = It->second;
    List.push_back(Inlined);
  }
}

LVList &getInlinedList(LVScope *Origin) {
  static LVList List;
  auto It = InlinedList.find(Origin);
  return (It == InlinedList.end()) ? List : It->second;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void dumpInlinedInfo(const char *Text, bool Full = false) {
  // Use 17 as the field length; it corresponds to '{InlinedFunction}'
  const unsigned LEN = 17;

  auto PrintEntry = [&](auto Text, LVScope *Scope) {
    std::stringstream SS;
    SS << Text << hexSquareString(Scope->getID())
       << hexSquareString(Scope->getParentScope()->getID()) << " "
       << std::setw(LEN) << std::left << formattedKind(Scope->kind());
    dbgs() << SS.str();
  };
  auto PrintExtra = [&](auto Text, LVScope *Scope) {
    dbgs() << Text;
    Scope->dumpCommon();
  };

  // For each origin scope prints its associated inlined scopes.
  dbgs() << "\nOrigin -> Inlined list: " << Text << "\n\n";
  for (auto &Entry : InlinedList) {
    LVScope *OriginScope = Entry.first;
    LVList &List = Entry.second;
    PrintEntry("", OriginScope);
    dbgs() << "\n";
    unsigned Count = 0;
    for (auto &Scope : List) {
      dbgs() << decString(++Count, /*Width=*/2);
      PrintEntry(" ", Scope);
      dbgs() << "\n";
    }
  }

  dbgs() << "\nOrigin -> Inlined: " << Text << "\n\n";
  for (auto &Entry : InlinedToOrigin) {
    LVScope *InlinedScope = Entry.first;
    LVScope *OriginScope = Entry.second;
    PrintEntry("", InlinedScope);
    dbgs() << " -> ";
    PrintEntry("", OriginScope);
    dbgs() << "\n";
  }

  if (Full) {
    dbgs() << "\n";
    for (auto &Entry : InlinedToOrigin) {
      LVScope *InlinedScope = Entry.first;
      LVScope *OriginScope = Entry.second;
      PrintExtra("OriginParent:  ", OriginScope->getParentScope());
      PrintExtra("Origin:        ", OriginScope);
      PrintExtra("InlinedParent: ", InlinedScope->getParentScope());
      PrintExtra("Inlined:       ", InlinedScope);
      dbgs() << "\n";
    }
  }
}
#endif

} // namespace

// These flavours of 'DINode's are not implemented but technically possible:
//   DW_TAG_APPLE_property   = 0x4200
//   DW_TAG_atomic_type      = 0x0047
//   DW_TAG_common_block     = 0x001a
//   DW_TAG_file_type        = 0x0029
//   DW_TAG_friend           = 0x002a
//   DW_TAG_generic_subrange = 0x0045
//   DW_TAG_immutable_type   = 0x004b
//   DW_TAG_module           = 0x001e
//   DW_TAG_variant_part     = 0x0033

// Create a logical element and setup the following information:
// - Name, DWARF tag, line
// - Collect any file information
LVElement *LVIRReader::constructElement(const DINode *DN) {
  dwarf::Tag Tag = DN->getTag();
  LVElement *Element = createElement(Tag);
  if (Element) {
    Element->setTag(Tag);
    addMD(DN, Element);

    if (StringRef Name = getMDName(DN); !Name.empty())
      Element->setName(Name);

    // Record any file information.
    if (const DIFile *File = getMDFile(DN))
      getOrCreateSourceID(File);
  }
  return Element;
}

void LVIRReader::setDefaultLowerBound(LVSourceLanguage *SL) {
  assert(SL && "Invalid language ID.");
  StringRef LanguageName = SL->getName();

  // Fortran uses 1 as the default lowerbound; other languages use 0.
  DefaultLowerBound = LanguageName.contains("fortran") ? 1 : 0;

  LLVM_DEBUG({ dbgs() << "Language Name: " << LanguageName << "\n"; });
}

bool LVIRReader::includeMinimalInlineScopes() const {
  return getCUNode()->getEmissionKind() == DICompileUnit::LineTablesOnly;
}

size_t LVIRReader::getOrCreateSourceID(const DIFile *File) {
  if (!File)
    return 0;

  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateSourceID]\n";
    dbgs() << "File: ";
    File->dump(TheModule);
  });
  addMD(File, CompileUnit);

  LLVM_DEBUG({
    dbgs() << "Directory: '" << File->getDirectory() << "'\n";
    dbgs() << "Filename:  '" << File->getFilename() << "'\n";
  });

  size_t FileIndex = getFileIndex(CompileUnit);
  auto [Iter, Inserted] = CompileUnitFiles.try_emplace(File, ++FileIndex);
  if (Inserted) {
    std::string Directory(File->getDirectory());
    if (Directory.empty())
      Directory = std::string(CompileUnit->getCompilationDirectory());

    std::string FullName;
    raw_string_ostream Out(FullName);
    Out << Directory << "/" << llvm::sys::path::filename(File->getFilename());
    CompileUnit->addFilename(transformPath(FullName));
    updateFileIndex(CompileUnit, FileIndex);
  } else {
    FileIndex = Iter->second;
  }

  LLVM_DEBUG({ dbgs() << "FileIndex: " << FileIndex << "\n"; });
  return FileIndex;
}

void LVIRReader::addSourceLine(LVElement *Element, unsigned Line,
                               const DIFile *File) {
  if (Line == 0)
    return;

  // After the scopes are created, the generic reader traverses the 'Children'
  // and performs additional setting tasks (resolve types names, references,
  // etc.). One of those tasks is select the correct string pool index based on
  // the commmand line options: --attribute=filename or --attribute=pathname.
  // As the 'Children' do not include logical lines, do that selection now,
  // by calling 'setFilename' if the logical element is a line.
  size_t FileID = getOrCreateSourceID(File);
  if (Element->getIsLine())
    Element->setFilename(CompileUnit->getFilename(FileID));
  else
    Element->setFilenameIndex(FileID);
  Element->setLineNumber(Line);

  LLVM_DEBUG({
    dbgs() << "\n[addSourceLine]\n";
    File->dump(TheModule);
    dbgs() << "FileIndex: " << Element->getFilenameIndex() << ", ";
    dbgs() << "ID:   " << Element->getID() << ", ";
    dbgs() << "Kind: " << Element->kind() << ", ";
    dbgs() << "Line: " << Element->getLineNumber() << ", ";
    dbgs() << "Name: " << Element->getName() << "\n";
  });
}

void LVIRReader::addSourceLine(LVElement *Element, const DIGlobalVariable *G) {
  assert(G);
  addSourceLine(Element, G->getLine(), G->getFile());
}

void LVIRReader::addSourceLine(LVElement *Element, const DIImportedEntity *IE) {
  assert(IE);
  addSourceLine(Element, IE->getLine(), IE->getFile());
}

void LVIRReader::addSourceLine(LVElement *Element, const DILabel *L) {
  assert(L);
  addSourceLine(Element, L->getLine(), L->getFile());
}

void LVIRReader::addSourceLine(LVElement *Element, const DILocalVariable *V) {
  assert(V);
  addSourceLine(Element, V->getLine(), V->getFile());
}

void LVIRReader::addSourceLine(LVElement *Element, const DILocation *DL) {
  assert(DL);
  addSourceLine(Element, DL->getLine(), DL->getFile());
}

void LVIRReader::addSourceLine(LVElement *Element, const DIObjCProperty *OP) {
  assert(OP);
  addSourceLine(Element, OP->getLine(), OP->getFile());
}

void LVIRReader::addSourceLine(LVElement *Element, const DISubprogram *SP) {
  assert(SP);
  addSourceLine(Element, SP->getLine(), SP->getFile());
}

void LVIRReader::addSourceLine(LVElement *Element, const DIType *Ty) {
  assert(Ty);
  addSourceLine(Element, Ty->getLine(), Ty->getFile());
}

void LVIRReader::addConstantValue(LVElement *Element,
                                  const DIExpression *DIExpr) {
  std::optional<DIExpression::SignedOrUnsignedConstant> Constant =
      DIExpr->isConstant();
  if (Constant == std::nullopt)
    return;
  std::stringstream Stream;
  uint64_t Value = DIExpr->getElement(1);
  if (DIExpression::SignedOrUnsignedConstant::SignedConstant == Constant) {
    if (int64_t SignedValue = static_cast<int64_t>(Value); SignedValue < 0) {
      Stream << "-";
      Value = static_cast<uint64_t>(-SignedValue);
    }
  }
  Stream << hexString(Value, 2);
  Element->setValue(Stream.str());
}

void LVIRReader::addConstantValue(LVElement *Element, const ConstantInt *CI,
                                  const DIType *Ty) {
  addConstantValue(Element, CI->getValue(), Ty);
}

void LVIRReader::addConstantValue(LVElement *Element, uint64_t Val,
                                  const DIType *Ty) {
  addConstantValue(Element, DebugHandlerBase::isUnsignedDIType(Ty), Val);
}

void LVIRReader::addConstantValue(LVElement *Element, uint64_t Val,
                                  bool Unsigned) {
  addConstantValue(Element, llvm::APInt(64, Val, Unsigned), Unsigned);
}

void LVIRReader::addConstantValue(LVElement *Element, const APInt &Val,
                                  const DIType *Ty) {
  addConstantValue(Element, Val, DebugHandlerBase::isUnsignedDIType(Ty));
}

void LVIRReader::addConstantValue(LVElement *Element, const APInt &Value,
                                  bool Unsigned) {
  SmallString<128> StringValue;
  Value.toString(StringValue, /*Radix=*/16, /*Signed=*/!Unsigned,
                 /*formatAsCLiteral=*/true, /*UpperCase=*/false,
                 /*InsertSeparators=*/false);
  Element->setValue(StringValue.str());
}

void LVIRReader::processLocationGaps() {
  if (options().getAttributeAnyLocation())
    for (LVSymbol *Symbol : SymbolsWithLocations)
      Symbol->fillLocationGaps();
}

void LVIRReader::processScopes() {
  // - Calculate their location ranges.
  // - Assign unique offset to the logical scopes, symbols and types,
  //   as the code the handles public names, expects them to have one.
  //   Use an arbitrary increment of 4.
  // - Resolve any line pattern match.
  // At this stage the compile unit and the root scopes they have the
  // same offset, which is incorrect. Update the compile unit offset.
  LVOffset Offset = OffsetIncrease;
  auto SetOffset = [&](LVElement *Element) {
    Element->setOffset(Offset);
    Offset += OffsetIncrease;
  };

  std::function<void(LVScope *)> TraverseScope = [&](LVScope *Current) {
    LVOffset Lower = Offset;
    SetOffset(Current);
    constructRange(Current);

    if (const LVScopes *Scopes = Current->getScopes())
      for (LVScope *Scope : *Scopes)
        TraverseScope(Scope);

    // Set an arbitrary, but strictly-increasing 'Offset' for symbols and types.
    if (const LVSymbols *Symbols = Current->getSymbols())
      for (LVSymbol *Symbol : *Symbols)
        SetOffset(Symbol);
    if (const LVTypes *Types = Current->getTypes())
      for (LVType *Type : *Types)
        SetOffset(Type);

    // Resolve any given pattern.
    if (const LVLines *Lines = Current->getLines())
      for (LVLine *Line : *Lines)
        patterns().resolvePatternMatch(Line);

    // Calculate contributions to the debug info.
    LVOffset Upper = Offset;
    if (options().getPrintSizes())
      CompileUnit->addSize(Current, Lower, Upper);
  };

  TraverseScope(CompileUnit);
}

std::string LVIRReader::getRegisterName(LVSmall Opcode,
                                        ArrayRef<uint64_t> Operands) {
  // At this point we are operating on a logical view item, with no access
  // to the underlying DWARF data used by LLVM.
  // We do not support DW_OP_regval_type here.
  if (Opcode == dwarf::DW_OP_regval_type)
    return {};

  if (Opcode == dwarf::DW_OP_regx || Opcode == dwarf::DW_OP_bregx) {
    // If the following trace is enabled, its output will be intermixed
    // with the logical view output, causing some confusion.
    // Leaving it here, just for any specific needs.
    // LLVM_DEBUG({
    //   dbgs() << "Printing Value: " << Operands[0] << " - "
    //          << ValueNameMap.getName(Operands[0]) << "\n";
    // });
    // Add an extra space for a better layout when printing locations.
    return " " + ValueNameMap.getName(Operands[0]);
  }

  llvm_unreachable("We shouldn't actually have any other reg types here!");
}

LVScope *LVIRReader::getParentScopeImpl(const DIScope *Context) {
  if (!Context)
    return CompileUnit;

  LLVM_DEBUG({
    dbgs() << "\n[getParentScopeImpl]\n";
    dbgs() << "Context: ";
    Context->dump(TheModule);
  });

  // Check for an already seen scope parent.
  if (LVScope *Parent = getScopeForSeenMD(Context))
    return Parent;

  // Traverse the scope hierarchy and construct the required scopes.
  return traverseParentScope(Context);
}

// Get the logical parent for the given metadata node.
LVScope *LVIRReader::getParentScope(const DILocation *DL) {
  assert(DL && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getParentScope]\n";
    dbgs() << "DL: ";
    DL->dump(TheModule);
  });

  return getParentScopeImpl(cast<DIScope>(DL->getScope()));
}

// Get the logical parent for the given metadata node.
LVScope *LVIRReader::getParentScope(const DINode *DN) {
  assert(DN && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getParentScope]\n";
    dbgs() << "DN: ";
    DN->dump(TheModule);
  });

  return getParentScopeImpl(getMDScope(DN));
}

LVScope *LVIRReader::traverseParentScope(const DIScope *Context) {
  if (!Context)
    return CompileUnit;

  LLVM_DEBUG({
    dbgs() << "\n[traverseParentScope]\n";
    dbgs() << "Context: \n";
    Context->dump(TheModule);
  });

  // Check if the metadata is already seen.
  if (LVScope *Parent = getScopeForSeenMD(Context))
    return Parent;

  // Create the scope parent.
  LVElement *Element = constructElement(Context);
  if (Element) {
    const DIScope *ParentContext = nullptr;
    if (const auto *SP = dyn_cast<DISubprogram>(Context)) {
      // Check for a specific 'Unit'.
      if (DICompileUnit *CU = SP->getUnit())
        ParentContext = getMDScope(SP->getDeclaration() ? CU : Context);
    } else {
      ParentContext = getMDScope(Context);
    }
    LVScope *Parent = traverseParentScope(ParentContext);
    if (Parent) {
      Parent->addElement(Element);
      constructScope(Element, Context);
    }
  }

  return static_cast<LVScope *>(Element);
}

// DW_TAG_base_type
//   DW_AT_name	("__ARRAY_SIZE_TYPE__")
//   DW_AT_byte_size	(0x08)
//   DW_AT_encoding	(DW_ATE_unsigned)
LVType *LVIRReader::getIndexType() {
  if (NodeIndexType)
    return NodeIndexType;

  // Construct an integer type to use for indexes.
  NodeIndexType = static_cast<LVType *>(createElement(dwarf::DW_TAG_base_type));
  if (NodeIndexType) {
    NodeIndexType->setIsFinalized();
    NodeIndexType->setName("__ARRAY_SIZE_TYPE__");
    CompileUnit->addElement(NodeIndexType);
  }

  return NodeIndexType;
}

void LVIRReader::addAccess(LVElement *Element, DINode::DIFlags Flags) {
  assert(Element && "Invalid logical element.");
  LLVM_DEBUG({
    dbgs() << "\n[addAccess]\n";
    dbgs() << "Flags: " << Flags << "\n";
  });

  const unsigned Accessibility = (Flags & DINode::FlagAccessibility);
  switch (Accessibility) {
  case DINode::FlagProtected:
    Element->setAccessibilityCode(dwarf::DW_ACCESS_protected);
    return;
  case DINode::FlagPrivate:
    Element->setAccessibilityCode(dwarf::DW_ACCESS_private);
    return;
  case DINode::FlagPublic:
    Element->setAccessibilityCode(dwarf::DW_ACCESS_public);
    return;
  case DINode::FlagZero:
    // If no explicit access control, provide the default for the parent.
    LVScope *Parent = Element->getParentScope();
    if (Parent->getIsClass()) {
      Element->setAccessibilityCode(dwarf::DW_ACCESS_private);
      return;
    }
    if (Parent->getIsStructure() || Parent->getIsUnion()) {
      Element->setAccessibilityCode(dwarf::DW_ACCESS_public);
      return;
    }
  }
}

// getFile()
//   DIScope
//   DILocation
//   DIVariable
//   DICommonBlock
//   DILabel
//   DIObjCProperty
//   DIImportedEntity
//   DIMacroFile
const DIFile *LVIRReader::getMDFile(const MDNode *MD) const {
  assert(MD && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getMDFile]\n";
    dbgs() << "MD: ";
    MD->dump(TheModule);
  });

  if (auto *T = dyn_cast<DIScope>(MD))
    return T->getFile();

  if (auto *T = dyn_cast<DILocation>(MD))
    return T->getFile();

  if (auto *T = dyn_cast<DIVariable>(MD))
    return T->getFile();

  if (auto *T = dyn_cast<DICommonBlock>(MD))
    return T->getFile();

  if (auto *T = dyn_cast<DILabel>(MD))
    return T->getFile();

  if (auto *T = dyn_cast<DIObjCProperty>(MD))
    return T->getFile();

  if (auto *T = dyn_cast<DIImportedEntity>(MD))
    return T->getFile();

  if (auto *T = dyn_cast<DIMacroFile>(MD))
    return T->getFile();

  return nullptr;
}

// getMDName()
//   DIScope
//   DIType
//   DISubprogram
//   DINamespace
//   DIModule
//   DITemplateParameter
//   DIVariable
//   DICommonBlock
//   DILabel
//   DIObjCProperty
//   DIImportedEntity
//   DIMacro
//   DIEnumerator
StringRef LVIRReader::getMDName(const DINode *DN) const {
  assert(DN && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getMDName]\n";
    dbgs() << "DN: ";
    DN->dump(TheModule);
  });

  if (auto *T = dyn_cast<DIImportedEntity>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DICompositeType>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DIDerivedType>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DILexicalBlockBase>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DIEnumerator>(DN))
    return T->getName();

  if (isa<DISubrange>(DN))
    return StringRef();

  if (auto *T = dyn_cast<DIVariable>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DIScope>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DITemplateParameter>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DILabel>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DIObjCProperty>(DN))
    return T->getName();

  if (auto *T = dyn_cast<DIMacro>(DN))
    return T->getName();

  assert((isa<DIFile>(DN) || isa<DICompileUnit>(DN)) && "Unhandled DINode.");
  return StringRef();
}

const DIScope *LVIRReader::getMDScope(const DINode *DN) const {
  assert(DN && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getMDScope]\n";
    dbgs() << "DN: ";
    DN->dump(TheModule);
  });

  if (dyn_cast<DIBasicType>(DN))
    return getCUNode();

  if (auto *T = dyn_cast<DINamespace>(DN)) {
    // The scope for global namespaces is nullptr.
    const DIScope *Context = T->getScope();
    if (!Context)
      Context = getCUNode();
    return Context;
  }

  if (auto *T = dyn_cast<DIImportedEntity>(DN))
    return T->getScope();

  if (auto *T = dyn_cast<DIVariable>(DN))
    return T->getScope();

  if (auto *T = dyn_cast<DIScope>(DN))
    return T->getScope();

  assert((isa<DIFile>(DN) || isa<DICompileUnit>(DN)) && "Unhandled DINode.");

  // Assume the scope to be the compile unit.
  return getCUNode();
}

//===----------------------------------------------------------------------===//
// Logical elements construction using IR metadata.
//===----------------------------------------------------------------------===//
void LVIRReader::addTemplateParams(LVElement *Element,
                                   const DINodeArray TParams) {
  assert(Element && "Invalid logical element");
  // assert(TParams && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[addTemplateParams]\n";
    for (const auto *Entry : TParams) {
      dbgs() << "Entry: ";
      Entry->dump(TheModule);
    }
  });

  // Add template parameters.
  for (const auto *Entry : TParams) {
    if (const auto *TTP = dyn_cast<DITemplateTypeParameter>(Entry))
      constructTemplateTypeParameter(Element, TTP);
    else if (const auto *TVP = dyn_cast<DITemplateValueParameter>(Entry))
      constructTemplateValueParameter(Element, TVP);
  }
}

// DISubprogram
void LVIRReader::applySubprogramAttributes(LVScope *Function,
                                           const DISubprogram *SP,
                                           bool SkipSPAttributes) {
  assert(Function && "Invalid logical element");
  assert(SP && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[applySubprogramAttributes]\n";
    dbgs() << "SP: ";
    SP->dump(TheModule);
  });

  // If -fdebug-info-for-profiling is enabled, need to emit the subprogram
  // and its source location.
  bool SkipSPSourceLocation =
      SkipSPAttributes && !getCUNode()->getDebugInfoForProfiling();
  if (!SkipSPSourceLocation)
    if (applySubprogramDefinitionAttributes(Function, SP, SkipSPAttributes))
      return;

  if (!SkipSPSourceLocation)
    addSourceLine(Function, SP);

  // Skip the rest of the attributes under -gmlt to save space.
  if (SkipSPAttributes)
    return;

  DITypeArray Args;
  if (const DISubroutineType *SPTy = SP->getType())
    Args = SPTy->getTypeArray();

  // Construct subprogram return type.
  if (Args.size()) {
    LVElement *ElementType = getOrCreateType(Args[0]);
    Function->setType(ElementType);
  }

  // Add virtuality info if available.
  Function->setVirtualityCode(SP->getVirtuality());

  if (!SP->isDefinition()) {
    // Add arguments. Do not add arguments for subprogram definition. They will
    // be handled while processing variables.
    constructSubprogramArguments(Function, Args);
  }

  if (SP->isArtificial())
    Function->setIsArtificial();

  if (!SP->isLocalToUnit())
    Function->setIsExternal();

  // Add accessibility info if available.
  addAccess(Function, SP->getFlags());
}

// DISubprogram
bool LVIRReader::applySubprogramDefinitionAttributes(LVScope *Function,
                                                     const DISubprogram *SP,
                                                     bool Minimal) {
  assert(Function && "Invalid logical element");
  assert(SP && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[applySubprogramDefinitionAttributes]\n";
    dbgs() << "SP: ";
    SP->dump(TheModule);
  });

  LVScope *Reference = nullptr;
  StringRef DeclLinkageName;
  if (const DISubprogram *SPDecl = SP->getDeclaration()) {
    if (!Minimal) {
      DITypeArray DeclArgs, DefinitionArgs;
      DeclArgs = SPDecl->getType()->getTypeArray();
      DefinitionArgs = SP->getType()->getTypeArray();

      // The element zero in 'DefinitionArgs' and 'DeclArgs' arrays is
      // the subprogram return type. A 'void' return does not have a
      // type and it is represented by a 'nullptr' value.
      // For the given test case and its IR:
      //
      // 1 struct Bar {
      // 2  bool foo(int a);
      // 3 };
      // 4
      // 5 bool Bar::foo(int a) {
      // 6   return false;
      // 7 }
      //
      // !10 = !DISubprogram(name: "foo", line: 5, type: !14,
      //                     spFlags: DISPFlagDefinition)
      // !13 = !DISubprogram(name: "foo", line: 2, type: !14, spFlags: 0)
      // !14 = !DISubroutineType(types: !15)
      // !15 = !{!16, !17, !18}
      // !16 = !DIBasicType(name: "bool", ...)
      //
      // '!15' represents both 'DefinitionArgs' and 'DeclArgs' arrays.
      // For cases where they have a different metadata node, use the
      // type from the 'DefinitionArgs' array as the correct type.
      if (DeclArgs.size() && DefinitionArgs.size())
        if (DefinitionArgs[0] != nullptr && DeclArgs[0] != DefinitionArgs[0]) {
          LVElement *ElementType = getOrCreateType(DefinitionArgs[0]);
          Function->setType(ElementType);
        }

      Reference = getScopeForSeenMD(SPDecl);
      assert(Reference && "Scope should've already been constructed.");
      // Look at the Decl's linkage name only if we emitted it.
      if (useAllLinkageNames())
        DeclLinkageName = SPDecl->getLinkageName();
      unsigned DeclID = getOrCreateSourceID(SPDecl->getFile());
      unsigned DefID = getOrCreateSourceID(SP->getFile());
      if (DeclID != DefID)
        Function->setFilenameIndex(DefID);

      if (SP->getLine() != SPDecl->getLine())
        Function->setLineNumber(SP->getLine());
    }
  }

  // Add function template parameters.
  addTemplateParams(Function, SP->getTemplateParams());

  // Add the linkage name if we have one and it isn't in the Decl.
  StringRef LinkageName = SP->getLinkageName();
  // Always emit it for abstract subprograms.
  if (DeclLinkageName != LinkageName && (useAllLinkageNames()))
    Function->setLinkageName(LinkageName);

  if (!Reference)
    return false;

  // Refer to the function declaration where all the other attributes
  // will be found.
  Function->setReference(Reference);
  Function->setHasReferenceSpecification();

  return true;
}

// DICompositeType
void LVIRReader::constructAggregate(LVScopeAggregate *Aggregate,
                                    const DICompositeType *CTy) {
  assert(Aggregate && "Invalid logical element");
  assert(CTy && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructAggregate]\n";
    dbgs() << "CTy: ";
    CTy->dump(TheModule);
  });

  if (Aggregate->getIsFinalized())
    return;
  Aggregate->setIsFinalized();

  dwarf::Tag Tag = Aggregate->getTag();

  // Add template parameters to a class, structure or union types.
  if (Tag == dwarf::DW_TAG_class_type || Tag == dwarf::DW_TAG_structure_type ||
      Tag == dwarf::DW_TAG_union_type)
    addTemplateParams(Aggregate, CTy->getTemplateParams());

  // Add elements to aggregate type.
  for (const auto *Member : CTy->getElements()) {
    if (!Member)
      continue;
    LLVM_DEBUG({
      dbgs() << "\nMember: ";
      Member->dump(TheModule);
    });
    if (const auto *SP = dyn_cast<DISubprogram>(Member))
      getOrCreateSubprogram(SP);
    else if (const DIDerivedType *DT = dyn_cast<DIDerivedType>(Member)) {
      dwarf::Tag Tag = Member->getTag();
      if (Tag == dwarf::DW_TAG_member || Tag == dwarf::DW_TAG_variable) {
        if (DT->isStaticMember())
          getOrCreateStaticMember(Aggregate, DT);
        else
          getOrCreateMember(Aggregate, DT);
      } else {
        getOrCreateType(DT, Aggregate);
      }
    }
  }
}

// DICompositeType
void LVIRReader::constructArray(LVScopeArray *Array,
                                const DICompositeType *CTy) {
  assert(Array && "Invalid logical element");
  assert(CTy && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructArray]\n";
    dbgs() << "CTy: ";
    CTy->dump(TheModule);
  });

  if (Array->getIsFinalized())
    return;
  Array->setIsFinalized();

  if (LVElement *BaseType = getOrCreateType(CTy->getBaseType()))
    Array->setType(BaseType);

  // Get an anonymous type for index type.
  LVType *IndexType = getIndexType();

  // Add subranges to array type.
  DINodeArray Entries = CTy->getElements();
  for (DINode *DN : Entries) {
    if (auto *SR = dyn_cast_or_null<DINode>(DN)) {
      if (SR->getTag() == dwarf::DW_TAG_subrange_type)
        constructSubrange(Array, cast<DISubrange>(SR), IndexType);
      else if (SR->getTag() == dwarf::DW_TAG_generic_subrange)
        constructGenericSubrange(Array, cast<DIGenericSubrange>(SR), IndexType);
    }
  }
}

// DICompositeType
void LVIRReader::constructEnum(LVScopeEnumeration *Enumeration,
                               const DICompositeType *CTy) {
  assert(Enumeration && "Invalid logical element");
  assert(CTy && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructEnum]\n";
    dbgs() << "CTy: ";
    CTy->dump(TheModule);
  });

  if (Enumeration->getIsFinalized())
    return;
  Enumeration->setIsFinalized();

  const DIType *Ty = CTy->getBaseType();
  bool IsUnsigned = Ty && DebugHandlerBase::isUnsignedDIType(Ty);

  if (LVElement *BaseType = getOrCreateType(Ty))
    Enumeration->setType(BaseType);

  if (CTy->getFlags() & DINode::FlagEnumClass)
    Enumeration->setIsEnumClass();

  // Add enumerators to enumeration type.
  DINodeArray Entries = CTy->getElements();
  for (const DINode *DN : Entries) {
    if (auto *Enum = dyn_cast_or_null<DIEnumerator>(DN)) {
      if (LVElement *Enumerator = constructElement(Enum)) {
        Enumerator->setIsFinalized();
        Enumeration->addElement(Enumerator);
        addConstantValue(Enumerator, Enum->getValue(), IsUnsigned);
      }
    }
  }
}

void LVIRReader::constructGenericSubrange(LVScopeArray *Array,
                                          const DIGenericSubrange *GSR,
                                          LVType *IndexType) {
  assert(Array && "Invalid logical element");
  assert(GSR && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructGenericSubrange]\n";
    dbgs() << "GSR: ";
    GSR->dump(TheModule);
  });

  LLVM_DEBUG({ dbgs() << "\nNot implemented\n"; });
}

// DIImportedEntity
void LVIRReader::constructImportedEntity(LVElement *Element,
                                         const DIImportedEntity *IE) {
  assert(Element && "Invalid logical element");
  assert(IE && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructImportedEntity]\n";
    dbgs() << "IE: ";
    IE->dump(TheModule);
  });

  if (LVElement *Import = constructElement(IE)) {
    Import->setIsFinalized();
    addSourceLine(Import, IE);
    LVScope *Parent = getParentScope(IE);
    Parent->addElement(Import);

    const DINode *Entity = IE->getEntity();
    LVElement *Target = getElementForSeenMD(Entity);
    if (!Target) {
      if (const auto *Ty = dyn_cast<DIType>(Entity))
        Target = getOrCreateType(Ty);
      else if (const auto *SP = dyn_cast<DISubprogram>(Entity))
        Target = getOrCreateSubprogram(SP);
      else if (const auto *NS = dyn_cast<DINamespace>(Entity))
        Target = getOrCreateNamespace(NS);
      else if (const auto *M = dyn_cast<DIModule>(Entity))
        Target = getOrCreateScope(M);
    }
    Import->setType(Target);
  }
}

// Traverse the 'inlinedAt' chain and create their associated inlined scopes.
LVScope *LVIRReader::getOrCreateInlinedScope(const DILocation *DL) {
  assert(DL && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateInlinedScope]\n";
    dbgs() << "DL: ";
    DL->dump(TheModule);
  });

  const DILocalScope *OriginContext = DL->getScope();
  LLVM_DEBUG({
    dbgs() << "OriginContext: ";
    OriginContext->dump(TheModule);
  });

  auto CreateScope = [&](const DILocalScope *Context) -> LVScope * {
    LVScope *Scope = nullptr;
    if (const auto *SP = dyn_cast<DISubprogram>(Context))
      Scope = getOrCreateSubprogram(SP);
    else
      Scope = getOrCreateScope(Context);
    LLVM_DEBUG({
      dbgs() << "Scope: ";
      Scope->dumpCommon();
    });

    return Scope;
  };

  const DILocation *InlinedAt = DL->getInlinedAt();
  if (!InlinedAt)
    return CreateScope(OriginContext);

  LLVM_DEBUG({
    dbgs() << "InlinedAt: ";
    InlinedAt->dump(TheModule);
  });

  // Check if the inlined scope is already created.
  if (LVScope *InlinedScope = getInlinedScope(OriginContext, InlinedAt))
    return InlinedScope;

  // Get or create the original context, which will be the seed for the
  // inlined scope that we intend to create.
  LVScope *OriginScope = CreateScope(OriginContext);

  dwarf::Tag Tag = OriginScope->getTag();
  if (OriginScope->getIsFunction() || OriginScope->getIsInlinedFunction()) {
    Tag = dwarf::DW_TAG_inlined_subroutine;
    OriginScope->setInlineCode(dwarf::DW_INL_inlined);
  }
  LVScope *InlinedScope = static_cast<LVScope *>(createElement(Tag));
  if (InlinedScope) {
    addInlinedScope(OriginContext, InlinedAt, InlinedScope);
    InlinedScope->setTag(Tag);
    InlinedScope->setIsFinalized();
    InlinedScope->setName(OriginScope->getName());
    InlinedScope->setType(OriginScope->getType());

    InlinedScope->setCallLineNumber(InlinedAt->getLine());
    InlinedScope->setCallFilenameIndex(
        getOrCreateSourceID(InlinedAt->getFile()));

    InlinedScope->setReference(OriginScope);
    InlinedScope->setHasReferenceAbstract();

    // Record the link between the origin and the inlined scope, to be
    // used to get the correct parent scope for logical lexical scopes.
    LLVM_DEBUG({
      dbgs() << "Linking\n";
      OriginScope->dumpCommon();
      InlinedScope->dumpCommon();
    });
    addInlinedInfo(OriginScope, InlinedScope);

    DILocalScope *AbstractContext = InlinedAt->getScope();
    LLVM_DEBUG({
      dbgs() << "AbstractContext: ";
      AbstractContext->dump(TheModule);
    });

    LVScope *AbstractScope = getOrCreateInlinedScope(InlinedAt);
    assert(AbstractScope && "Logical scope is NULL.");
    LLVM_DEBUG({
      dbgs() << "AbstractScope: ";
      AbstractScope->dumpCommon();
    });

    // Add the created inlined scope.
    AbstractScope->addElement(InlinedScope);

    LLVM_DEBUG({
      dbgs() << "InlinedScope:  ";
      InlinedScope->dumpCommon();
    });
  }

  return InlinedScope;
}

LVScope *LVIRReader::getOrCreateAbstractScope(const DILocation *DL) {
  assert(DL && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateAbstractScope]\n";
    dbgs() << "DL: ";
    DL->dump(TheModule);
  });

  // Create the 'inlined' scope.
  LVScope *InlinedScope = getOrCreateInlinedScope(DL);
  assert(InlinedScope && "InlinedScope is null.");
  return InlinedScope;
}

void LVIRReader::constructLine(LVScope *Scope, const DISubprogram *SP,
                               Instruction &I,
                               bool &GenerateLineBeforePrologue) {
  assert(Scope && "Invalid logical element");
  assert(SP && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructLine]\n";
    dbgs() << "Instruction: ";
    I.dump();
    dbgs() << "Logical Scope: ";
    Scope->dumpCommon();
  });

  auto AddDebugLine = [&](LVScope *Parent, unsigned ID) -> LVLine * {
    assert(Parent && "Invalid logical element");
    assert(ID == Metadata::DILocationKind && "Invalid Metadata Object");
    LLVM_DEBUG({
      dbgs() << "\n[AddDebugLine]\n";
      dbgs() << "Parent: ";
      Parent->dumpCommon();
    });

    LVLine *Line = createLineDebug();
    if (Line) {
      Parent->addElement(Line);

      Line->setIsFinalized();
      Line->setAddress(CurrentOffset);

      // FIXME: How to get discrimination flags:
      // IsStmt, BasicBlock, EndSequence, EpilogueBegin, PrologueEnd.
      //
      // Explore the 'Key Instructions' information added to the metadata:
      //   !DILocation(line: ..., scope: ..., atomGroup: ..., atomRank: ...)

      // Add mapping for this debug line.
      CompileUnit->addMapping(Line, /*SectionIndex=*/0);

      // Replicate the DWARF reader functionality of adding a linkage
      // name to a function with ranges (logical lines), regardless if
      // the declaration has already one.
      if (!Parent->getLinkageNameIndex() &&
          Parent->getHasReferenceSpecification()) {
        Parent->setLinkageName(Parent->getReference()->getLinkageName());
      }
      GenerateLineBeforePrologue = false;
    }

    return Line;
  };

  auto AddAssemblerLine = [&](LVScope *Parent) {
    assert(Parent && "Invalid logical element");

    static const char *WhiteSpace = " \t\n\r\f\v";
    static std::string Metadata("metadata ");

    auto RemoveAll = [](std::string &Input, std::string &Pattern) {
      std::string::size_type Len = Pattern.length();
      for (std::string::size_type Index = Input.find(Pattern);
           Index != std::string::npos; Index = Input.find(Pattern))
        Input.erase(Index, Len);
    };

    std::string InstructionText;
    raw_string_ostream Stream(InstructionText);
    Stream << I;
    // Remove the 'metadata ' pattern from the instruction text.
    RemoveAll(InstructionText, Metadata);
    std::string_view Text(InstructionText);
    const auto pos(Text.find_first_not_of(WhiteSpace));
    Text.remove_prefix(std::min(pos, Text.length()));

    // Create an instruction line at the given scope.
    if (LVLineAssembler *Line = createLineAssembler()) {
      Line->setIsFinalized();
      Line->setAddress(CurrentOffset);
      Line->setName(Text);
      Parent->addElement(Line);
    }
  };

  LVScope *Parent = Scope;
  if (const DebugLoc DbgLoc = I.getDebugLoc()) {
    const DILocation *DL = DbgLoc.get();
    LLVM_DEBUG({
      dbgs() << "DL: ";
      DL->dump(TheModule);
    });

    Parent = getOrCreateAbstractScope(DL);
    assert(Parent && "Invalid logical element");
    LLVM_DEBUG({
      dbgs() << "Parent: ";
      Parent->dumpCommon();
    });

    if (options().getPrintLines() && DL->getLine()) {
      if (LVLine *Line = AddDebugLine(Parent, DL->getMetadataID())) {
        addMD(DL, Line);
        addSourceLine(Line, DL);
        GenerateLineBeforePrologue = false;
      }
    }
  }

  // Generate a logical line before the function prologue.
  if (options().getPrintLines() && GenerateLineBeforePrologue) {
    if (LVLine *Line = AddDebugLine(Parent, Metadata::DILocationKind)) {
      addSourceLine(Line, SP);
      GenerateLineBeforePrologue = false;
    }
  }

  // Create assembler line.
  if (options().getPrintInstructions())
    AddAssemblerLine(Parent);
}

LVSymbol *LVIRReader::getOrCreateMember(LVScope *Aggregate,
                                        const DIDerivedType *DT) {
  assert(Aggregate && "Invalid logical element");
  assert(DT && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateMember]\n";
    dbgs() << "DT: ";
    DT->dump(TheModule);
  });

  LVSymbol *Member = getSymbolForSeenMD(DT);
  if (Member && Member->getIsFinalized())
    return Member;

  if (!options().getPrintSymbols()) {
    // Just create the symbol type.
    getOrCreateType(DT->getBaseType());
    return nullptr;
  }

  if (!Member)
    Member = static_cast<LVSymbol *>(getOrCreateType(DT, Aggregate));
  if (Member) {
    Member->setIsFinalized();
    addSourceLine(Member, DT);
    if (DT->getTag() == dwarf::DW_TAG_inheritance && DT->isVirtual()) {
      Member->addLocation(dwarf::DW_AT_data_member_location, /*LowPC=*/0,
                          /*HighPC=*/-1, /*SectionOffset=*/0,
                          /*OffsetOnEntry=*/0);
    } else {
      uint64_t OffsetInBytes = 0;

      bool IsBitfield = DT->isBitField();
      if (IsBitfield) {
        Member->setBitSize(DT->getSizeInBits());
      } else {
        // This is not a bitfield.
        OffsetInBytes = DT->getOffsetInBits() / 8;
      }

      if (DwarfVersion <= 2) {
        // DW_AT_data_member_location:
        //   DW_FORM_data1, DW_OP_plus_uconst, DW_FORM_udata, OffsetInBytes
        Member->addLocation(dwarf::DW_AT_data_member_location, /*LowPC=*/0,
                            /*HighPC=*/-1, /*SectionOffset=*/0,
                            /*OffsetOnEntry=*/0);
        Member->addLocationOperands(dwarf::DW_OP_plus_uconst, {OffsetInBytes});
      } else if (!IsBitfield || DwarfVersion < 4) {
        // DW_AT_data_member_location:
        //   DW_FORM_udata, OffsetInBytes
        Member->addLocationConstant(dwarf::DW_AT_data_member_location,
                                    OffsetInBytes,
                                    /*OffsetOnEntry=*/0);
      }
    }
  }

  // Add accessibility info if available.
  if (!DT->isStaticMember())
    addAccess(Member, DT->getFlags());

  if (DT->isVirtual())
    Member->setVirtualityCode(dwarf::DW_VIRTUALITY_virtual);

  if (DT->isArtificial())
    Member->setIsArtificial();

  return Member;
}

// DIBasicType
// DICommonBlock
// DICompileUnit
// DICompositeType
// DIDerivedType
// DIFile
// DILexicalBlock
// DILexicalBlockFile
// DIModule
// DINamespace
// DISubprogram
// DISubroutineType
// DIStringType
void LVIRReader::constructScope(LVElement *Element, const DIScope *Context) {
  assert(Element && "Invalid logical element");
  assert(Context && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructScope]\n";
    dbgs() << "Context: ";
    Context->dump(TheModule);
  });

  if (const DICompositeType *CTy =
          dyn_cast_if_present<DICompositeType>(Context)) {
    constructType(static_cast<LVScope *>(Element), CTy);
  } else if (const DIDerivedType *DT =
                 dyn_cast_if_present<DIDerivedType>(Context)) {
    constructType(Element, DT);
  } else if (const DISubprogram *SP =
                 dyn_cast_if_present<DISubprogram>(Context)) {
    getOrCreateSubprogram(static_cast<LVScope *>(Element), SP);
  } else if (dyn_cast_if_present<DINamespace>(Context)) {
    Element->setIsFinalized();
  } else if (dyn_cast_if_present<DILexicalBlock>(Context)) {
    Element->setIsFinalized();
  }
}

LVSymbol *LVIRReader::getOrCreateStaticMember(LVScope *Aggregate,
                                              const DIDerivedType *DT) {
  assert(Aggregate && "Invalid logical element");
  assert(DT && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateStaticMember]\n";
    dbgs() << "DT: ";
    DT->dump(TheModule);
  });

  LVSymbol *Member = getSymbolForSeenMD(DT);
  if (Member && Member->getIsFinalized())
    return Member;

  if (!options().getPrintSymbols()) {
    // Just create the symbol type.
    getOrCreateType(DT->getBaseType());
    return nullptr;
  }

  if (!Member)
    Member = static_cast<LVSymbol *>(getOrCreateType(DT, Aggregate));
  if (Member) {
    Member->setIsFinalized();
    addSourceLine(Member, DT);
    Member->setIsExternal();
  }

  return Member;
}

// DISubprogram
LVScope *LVIRReader::getOrCreateSubprogram(const DISubprogram *SP) {
  assert(SP && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateSubprogram]\n";
    dbgs() << "SP: ";
    SP->dump(TheModule);
  });

  LVScope *Function = getScopeForSeenMD(SP);
  if (Function && Function->getIsFinalized())
    return Function;

  if (!Function)
    Function = static_cast<LVScope *>(constructElement(SP));
  if (Function) {
    // For both member functions (declaration and definition) its parent
    // is the containing class. The 'definition' points back to its
    // 'declaration' via the 'getDeclaration' return value.
    LVScope *Parent = SP->getDeclaration()
                          ? SP->isLocalToUnit() || SP->isDefinition()
                                ? CompileUnit
                                : getParentScope(SP)->getParentScope()
                          : getParentScope(SP);
    // The 'getParentScope' traverses the scope hierarchy and it creates
    // the scope chain and any associated types.
    // Check that the 'Function' is not already in the parent.
    if (!Function->getParent())
      Parent->addElement(Function);

    getOrCreateSubprogram(Function, SP, includeMinimalInlineScopes());
  }

  return Function;
}

// DISubprogram
LVScope *LVIRReader::getOrCreateSubprogram(LVScope *Function,
                                           const DISubprogram *SP,
                                           bool Minimal) {
  assert(Function && "Invalid logical element");
  assert(SP && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateSubprogram]\n";
    dbgs() << "SP: ";
    SP->dump(TheModule);
  });

  if (Function->getIsFinalized())
    return Function;
  Function->setIsFinalized();

  // Get 'declaration' node in order to generate the DW_AT_specification.
  if (const DISubprogram *SPDecl = SP->getDeclaration()) {
    if (!Minimal) {
      // Build the declaration now to ensure it precedes the definition.
      getOrCreateSubprogram(SPDecl);
    }
  }

  // Check for additional retained nodes.
  for (const DINode *DN : SP->getRetainedNodes()) {
    if (const auto *IE = dyn_cast<DIImportedEntity>(DN))
      constructImportedEntity(Function, IE);
    else if (const auto *TTP = dyn_cast<DITemplateTypeParameter>(DN))
      constructTemplateTypeParameter(Function, TTP);
    else if (const auto *TVP = dyn_cast<DITemplateValueParameter>(DN))
      constructTemplateValueParameter(Function, TVP);
  }

  applySubprogramAttributes(Function, SP);

  // Check if we are dealing with the Global Init/Cleanup Function.
  if (SP->isArtificial() && SP->isLocalToUnit() && SP->isDefinition() &&
      SP->getName().empty())
    Function->setName(SP->getLinkageName());

  return Function;
}

void LVIRReader::constructSubprogramArguments(LVScope *Function,
                                              const DITypeArray Args) {
  assert(Function && "Invalid logical element");
  LLVM_DEBUG({
    dbgs() << "\n[constructSubprogramArguments]\n";
    for (unsigned i = 1, N = Args.size(); i < N; ++i) {
      if (const DIType *Ty = Args[i]) {
        dbgs() << "Ty: ";
        Ty->dump(TheModule);
      }
    }
  });

  for (unsigned I = 1, N = Args.size(); I < N; ++I) {
    const DIType *Ty = Args[I];
    LVElement *Parameter = nullptr;
    if (Ty) {
      // Create a formal parameter.
      LVElement *ParameterType = getOrCreateType(Ty);
      Parameter = createElement(dwarf::DW_TAG_formal_parameter);
      if (Parameter) {
        Parameter->setType(ParameterType);
        if (Ty->isArtificial())
          Parameter->setIsArtificial();
      }
    } else {
      // Add an unspecified parameter.
      Parameter = createElement(dwarf::DW_TAG_unspecified_parameters);
    }
    if (Parameter) {
      Function->addElement(Parameter);
      Parameter->setIsFinalized();
    }
  }
}

// DISubrange
void LVIRReader::constructSubrange(LVScopeArray *Array, const DISubrange *SR,
                                   LVType *IndexType) {
  assert(Array && "Invalid logical element");
  assert(SR && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructSubrange]\n";
    dbgs() << "SR: ";
    SR->dump(TheModule);
  });

  // The DISubrange can be shared between different arrays, when they are
  // the same. We need to create independent logical elements for each one,
  // as they are going to be added to different arrays.
  if (LVTypeSubrange *Subrange =
          static_cast<LVTypeSubrange *>(constructElement(SR))) {
    Subrange->setIsFinalized();
    Array->addElement(Subrange);
    Subrange->setType(IndexType);

    int64_t Count = -1;
    // If Subrange has a Count field, use it.
    // Otherwise, if it has an upperboud, use (upperbound - lowerbound + 1),
    // where lowerbound is from the LowerBound field of the Subrange,
    // or the language default lowerbound if that field is unspecified.
    if (auto *CI = dyn_cast_if_present<ConstantInt *>(SR->getCount()))
      Count = CI->getSExtValue();
    else if (auto *UI =
                 dyn_cast_if_present<ConstantInt *>(SR->getUpperBound())) {
      // Fortran uses 1 as the default lowerbound; other languages use 0.
      int64_t Lowerbound = getDefaultLowerBound();
      auto *LI = dyn_cast_if_present<ConstantInt *>(SR->getLowerBound());
      Lowerbound = (LI) ? LI->getSExtValue() : Lowerbound;
      Count = UI->getSExtValue() - Lowerbound + 1;
    }

    if (Count == -1)
      Count = 0;
    Subrange->setCount(Count);
  }
}

// DITemplateTypeParameter
void LVIRReader::constructTemplateTypeParameter(
    LVElement *Element, const DITemplateTypeParameter *TTP) {
  assert(Element && "Invalid logical element");
  assert(TTP && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructTemplateTypeParameter]\n";
    dbgs() << "TTP: ";
    TTP->dump(TheModule);
  });

  // The DITemplateTypeParameter can be shared between different subprogram
  // in their DITemplateParameterArray describing the template parameters.
  // We need to create independent logical elements for each one, as they are
  // going to be added to different function.
  if (LVElement *Parameter = constructElement(TTP)) {
    Parameter->setIsFinalized();
    // Add element to parent (always the given Element).
    LVScope *Parent = static_cast<LVScope *>(Element);
    Parent->addElement(Parameter);
    // Mark the parent as template.
    Parent->setIsTemplate();

    // Add the type if it exists, it could be void and therefore no type.
    if (const DIType *Ty = TTP->getType()) {
      LVElement *Type = getElementForSeenMD(Ty);
      if (!Type)
        Type = getOrCreateType(Ty);
      Parameter->setType(Type);
    }
  }
}

// DITemplateValueParameter
void LVIRReader::constructTemplateValueParameter(
    LVElement *Element, const DITemplateValueParameter *TVP) {
  assert(Element && "Invalid logical element");
  assert(TVP && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructTemplateValueParameter]\n";
    dbgs() << "TVP: ";
    TVP->dump(TheModule);
  });

  // The DITemplateValueParameter can be shared between different subprogram
  // in their DITemplateParameterArray describing the template parameters.
  // We need to create independent logical elements for each one, as they are
  // going to be added to different function.
  if (LVElement *Parameter = constructElement(TVP)) {
    Parameter->setIsFinalized();
    // Add element to parent (always the given Element).
    LVScope *Parent = static_cast<LVScope *>(Element);
    Parent->addElement(Parameter);
    // Mark the parent as template.
    Parent->setIsTemplate();

    // Add the type if there is one, template template and template parameter
    // packs will not have a type.
    if (TVP->getTag() == dwarf::DW_TAG_template_value_parameter) {
      LVElement *Type = getOrCreateType(TVP->getType());
      Parameter->setType(Type);
    }
    if (Metadata *Value = TVP->getValue()) {
      if (ConstantInt *CI = mdconst::dyn_extract<ConstantInt>(Value))
        addConstantValue(Parameter, CI, TVP->getType());
      else if (GlobalValue *GV = mdconst::dyn_extract<GlobalValue>(Value)) {
        // We cannot describe the location of dllimport'd entities: the
        // computation of their address requires loads from the IAT.
        if (!GV->hasDLLImportStorageClass()) {
        }
      } else if (TVP->getTag() == dwarf::DW_TAG_GNU_template_template_param) {
        assert(isa<MDString>(Value));
        // Add the value for dwarf::DW_AT_GNU_template_name.
        Parameter->setValue(cast<MDString>(Value)->getString());
      } else if (TVP->getTag() == dwarf::DW_TAG_GNU_template_parameter_pack) {
        addTemplateParams(Parameter, cast<MDTuple>(Value));
      }
    }
  }
}

// DICompositeType
//   DW_TAG_array_type
//   DW_TAG_class_type
//   DW_TAG_enumeration_type
//   DW_TAG_structure_type
//   DW_TAG_union_type
void LVIRReader::constructType(LVScope *Scope, const DICompositeType *CTy) {
  assert(Scope && "Invalid logical element");
  assert(CTy && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructType]\n";
    dbgs() << "CTy: ";
    CTy->dump(TheModule);
  });

  dwarf::Tag Tag = Scope->getTag();
  switch (Tag) {
  case dwarf::DW_TAG_array_type:
    constructArray(static_cast<LVScopeArray *>(Scope), CTy);
    break;
  case dwarf::DW_TAG_enumeration_type:
    constructEnum(static_cast<LVScopeEnumeration *>(Scope), CTy);
    break;
  // FIXME: Not implemented.
  case dwarf::DW_TAG_variant_part:
  case dwarf::DW_TAG_namelist:
    break;
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_class_type: {
    constructAggregate(static_cast<LVScopeAggregate *>(Scope), CTy);
    break;
  }
  default:
    break;
  }

  if (Tag == dwarf::DW_TAG_enumeration_type ||
      Tag == dwarf::DW_TAG_class_type || Tag == dwarf::DW_TAG_structure_type ||
      Tag == dwarf::DW_TAG_union_type) {
    // Add accessibility info if available.
    addAccess(Scope, CTy->getFlags());

    // Add source line info if available.
    if (!CTy->isForwardDecl())
      addSourceLine(Scope, CTy);
  }
}

// DIDerivedType
//   DW_TAG_atomic_type
//   DW_TAG_const_type
//   DW_TAG_friend
//   DW_TAG_inheritance
//   DW_TAG_member
//   DW_TAG_immutable_type
//   DW_TAG_pointer_type
//   DW_TAG_ptr_to_member_type
//   DW_TAG_reference_type
//   DW_TAG_restrict_type
//   DW_TAG_typedef
//   DW_TAG_volatile_type
void LVIRReader::constructType(LVElement *Element, const DIDerivedType *DT) {
  assert(Element && "Invalid logical element");
  assert(DT && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructType]\n";
    dbgs() << "DT: ";
    DT->dump(TheModule);
  });

  // For DW_TAG_member, the flag is set during the construction of the
  // aggregate type (DICompositeType).
  if (DT->getTag() != dwarf::DW_TAG_member)
    Element->setIsFinalized();

  LVElement *BaseType = getOrCreateType(DT->getBaseType());
  Element->setType(BaseType);

  // Add accessibility info if available.
  if (!DT->isStaticMember())
    addAccess(Element, DT->getFlags());

  if (DT->isVirtual())
    Element->setVirtualityCode(dwarf::DW_VIRTUALITY_virtual);

  if (DT->isArtificial())
    Element->setIsArtificial();

  // Add source line info if available and TyDesc is not a forward declaration.
  if (!DT->isForwardDecl())
    addSourceLine(Element, DT);
}

// DISubroutineType
void LVIRReader::constructType(LVScope *Function,
                               const DISubroutineType *SPTy) {
  assert(Function && "Invalid logical element");
  assert(SPTy && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[constructType]\n";
    dbgs() << "SPTy: ";
    SPTy->dump(TheModule);
  });

  if (Function->getIsFinalized())
    return;
  Function->setIsFinalized();

  // For DISubprogram, the DISubroutineType contains the types for:
  //   return type, param 1 type, ..., param n type
  DITypeArray Args = SPTy->getTypeArray();
  if (Args.size()) {
    LVElement *ElementType = getOrCreateType(Args[0]);
    Function->setType(ElementType);
  }

  constructSubprogramArguments(Function, Args);
}

// DINamespace
LVScope *LVIRReader::getOrCreateNamespace(const DINamespace *NS) {
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateNamespace]\n";
    dbgs() << "NS: ";
    NS->dump(TheModule);
  });

  LVScope *Scope = getOrCreateScope(NS);
  if (Scope) {
    StringRef Name = NS->getName();
    if (Name.empty())
      Scope->setName("(anonymous namespace)");
  }

  return Scope;
}

LVScope *LVIRReader::getOrCreateScope(const DIScope *Context) {
  assert(Context && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateScope]\n";
    dbgs() << "Context: ";
    Context->dump(TheModule);
  });

  // Check if the scope is already created.
  LVScope *Scope = getScopeForSeenMD(Context);
  if (Scope)
    return Scope;

  Scope = static_cast<LVScope *>(constructElement(Context));
  if (Scope) {
    // Add element to parent.
    LVScope *Parent = getParentScope(Context);
    Parent->addElement(Scope);
  }

  return Scope;
}

// DICompositeType
// DIDerivedType
// DISubroutineType
LVElement *LVIRReader::getOrCreateType(const DIType *Ty, LVScope *Scope) {
  if (!Ty)
    return nullptr;

  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateType]\n";
    dbgs() << "Ty :";
    Ty->dump(TheModule);
  });

  // Check if the element is already created.
  LVElement *Element = getElementForSeenMD(Ty);
  if (Element)
    return Element;

  Element = constructElement(Ty);
  if (Element) {
    // Add element to parent.
    LVScope *Parent = Scope ? Scope : getParentScope(Ty);
    Parent->addElement(Element);

    if (isa<DIBasicType>(Ty)) {
      Element->setIsFinalized();
    } else if (const DIDerivedType *DT = dyn_cast<DIDerivedType>(Ty)) {
      constructType(Element, DT);
    } else if (const DICompositeType *CTy = dyn_cast<DICompositeType>(Ty)) {
      constructType(static_cast<LVScope *>(Element), CTy);
    } else if (const DISubroutineType *SPTy = dyn_cast<DISubroutineType>(Ty)) {
      constructType(static_cast<LVScope *>(Element), SPTy);
    }
  }

  return Element;
}

// DIGlobalVariableExpression
LVSymbol *
LVIRReader::getOrCreateVariable(const DIGlobalVariableExpression *GVE) {
  assert(GVE && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateVariable]\n";
    dbgs() << "GVE: ";
    GVE->dump(TheModule);
  });

  const DIGlobalVariable *DIGV = GVE->getVariable();
  LVSymbol *Symbol = getSymbolForSeenMD(DIGV);
  if (!Symbol)
    Symbol = getOrCreateVariable(DIGV);

  if (Symbol) {
    // Add location and operation entries.
    Symbol->addLocation(dwarf::DW_AT_location, /*LowPC=*/0, /*HighPC=*/-1,
                        /*SectionOffset=*/0, /*OffsetOnEntry=*/0);
    Symbol->addLocationOperands(dwarf::DW_OP_addrx, PoolAddressIndex++);
    if (const DIExpression *DIExpr = GVE->getExpression())
      addConstantValue(Symbol, DIExpr);
  }
  return Symbol;
}

LVSymbol *LVIRReader::getOrCreateInlinedVariable(LVSymbol *OriginSymbol,
                                                 const DILocation *DL) {
  assert(OriginSymbol && "Invalid logical element");
  assert(DL && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateInlinedVariable]\n";
    dbgs() << "DL: ";
    DL->dump(TheModule);
  });

  const DILocation *InlinedAt = DL->getInlinedAt();
  if (!InlinedAt) {
    return nullptr;
  }

  dwarf::Tag Tag = OriginSymbol->getTag();
  LVSymbol *InlinedSymbol = static_cast<LVSymbol *>(createElement(Tag));
  if (InlinedSymbol) {
    InlinedSymbol->setTag(Tag);
    InlinedSymbol->setIsFinalized();
    InlinedSymbol->setName(OriginSymbol->getName());
    InlinedSymbol->setType(OriginSymbol->getType());

    InlinedSymbol->setCallLineNumber(InlinedAt->getLine());
    InlinedSymbol->setCallFilenameIndex(
        getOrCreateSourceID(InlinedAt->getFile()));

    OriginSymbol->setInlineCode(dwarf::DW_INL_inlined);
    InlinedSymbol->setReference(OriginSymbol);
    InlinedSymbol->setHasReferenceAbstract();

    if (OriginSymbol->getIsParameter())
      InlinedSymbol->setIsParameter();

    // Get or create the local scope associated with the location.
    LVScope *InlinedScope = getOrCreateInlinedScope(DL);
    assert(InlinedScope && "Invalid logical element");

    // Add the created inlined scope.
    InlinedScope->addElement(InlinedSymbol);
  }

  return InlinedSymbol;
}

// DIGlobalVariable
// DILocalVariable
LVSymbol *LVIRReader::getOrCreateVariable(const DIVariable *Var,
                                          const DILocation *DL) {
  assert(Var && "Invalid metadata node.");
  LLVM_DEBUG({
    dbgs() << "\n[getOrCreateVariable]\n";
    dbgs() << "Var: ";
    Var->dump(TheModule);
    if (DL) {
      dbgs() << "DL: ";
      DL->dump(TheModule);
    }
  });

  // Use the 'InlinedAt' information to identify a symbol that is being
  // inlined. Its abstract representation is created just once.
  const DILocation *InlinedAt = DL ? DL->getInlinedAt() : nullptr;

  LVSymbol *Symbol = getSymbolForSeenMD(Var);
  if (Symbol && Symbol->getIsFinalized() && !InlinedAt)
    return Symbol;

  if (!options().getPrintSymbols()) {
    // Just create the symbol type.
    getOrCreateType(Var->getType());
    if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(Var)) {
      if (MDTuple *TP = GV->getTemplateParams())
        addTemplateParams(Symbol, DINodeArray(TP));
    }
    return nullptr;
  }

  if (!Symbol)
    Symbol = static_cast<LVSymbol *>(constructElement(Var));
  if (Symbol && !Symbol->getIsFinalized()) {
    Symbol->setIsFinalized();
    LVScope *Parent = getParentScope(Var);
    Parent->addElement(Symbol);

    Symbol->setName(Var->getName());

    // Create symbol type.
    if (LVElement *SymbolType = getOrCreateType(Var->getType()))
      Symbol->setType(SymbolType);

    if (const DILocalVariable *LV = dyn_cast<DILocalVariable>(Var)) {
      // Add line number info.
      addSourceLine(Symbol, LV);
      if (LV->isParameter()) {
        Symbol->setIsParameter();
        if (LV->isArtificial())
          Symbol->setIsArtificial();
      }
    } else {
      const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(Var);
      if (useAllLinkageNames())
        Symbol->setLinkageName(GV->getLinkageName());

      // Get 'declaration' node in order to generate the DW_AT_specification.
      if (const DIDerivedType *GVDecl = GV->getStaticDataMemberDeclaration()) {
        LVSymbol *Reference = static_cast<LVSymbol *>(getOrCreateType(GVDecl));
        if (Reference) {
          Symbol->setReference(Reference);
          Symbol->setHasReferenceSpecification();
        }
      } else {
        if (!GV->isLocalToUnit())
          Symbol->setIsExternal();
        // Add line number info.
        addSourceLine(Symbol, GV);
      }

      if (MDTuple *TP = GV->getTemplateParams())
        addTemplateParams(Symbol, DINodeArray(TP));
    }
  }

  // Create the 'inlined' symbol.
  if (DL)
    getOrCreateInlinedVariable(Symbol, DL);

  return Symbol;
}

#ifdef LLVM_DEBUG
void LVIRReader::printAllInstructions(BasicBlock *BB) {
  const Function *F = BB->getParent();
  if (!F)
    return;
  const DISubprogram *SP = cast<DISubprogram>(F->getSubprogram());
  LLVM_DEBUG({
    dbgs() << "\nBegin all instructions: '" << SP->getName() << "'\n";
    for (Instruction &I : *BB) {
      dbgs() << "I: '" << I << "'\n";
      for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
        dbgs() << "  Var: ";
        DVR.getVariable()->dump(TheModule);
      }
      if (const auto *DL =
              cast_or_null<DILocation>(I.getMetadata(LLVMContext::MD_dbg))) {
        dbgs() << "  DL: ";
        DL->dump(TheModule);
      }
    }
    dbgs() << "End all instructions: '" << SP->getName() << "'\n\n";
  });
}
#endif

void LVIRReader::processBasicBlocks(Function &F) {
  const DISubprogram *SP = cast_or_null<DISubprogram>(F.getSubprogram());
  if (!SP)
    return;

  LLVM_DEBUG({
    dbgs() << "\n[processBasicBlocks]\n";
    dbgs() << "SP: ";
    SP->dump(TheModule);
  });

  // Check if we need to add a dwarf::DW_TAG_unspecified_parameters.
  bool AddUnspecifiedParameters = false;
  if (const DISubroutineType *SPTy = SP->getType()) {
    DITypeArray Args = SPTy->getTypeArray();
    unsigned N = Args.size();
    if (N > 1) {
      const DIType *Ty = Args[N - 1];
      if (!Ty)
        AddUnspecifiedParameters = true;
    }
  }

  LVScope *Scope = getOrCreateSubprogram(SP);

  SmallVector<DebugVariableAggregate> SeenVars;

  // Handle dbg.values and dbg.declare.
  auto HandleDbgVariable = [&](auto *DbgVar) {
    LLVM_DEBUG({
      dbgs() << "\n[HandleDbgVariable]\n";
      dbgs() << "DbgVar: ";
      DbgVar->dump();
    });

    DebugVariableAggregate DVA(DbgVar);
    if (!DbgValueRanges->hasVariableEntry(DVA)) {
      DbgValueRanges->addVariable(&F, DVA);
      SeenVars.push_back(DVA);
    }

    // Skip undefined values.
    if (!DbgVar->isKillLocation())
      getOrCreateVariable(DbgVar->getVariable(), DbgVar->getDebugLoc().get());
  };

  // Generate logical debug line before prologue.
  bool GenerateLineBeforePrologue = true;
  for (BasicBlock &BB : F) {
    printAllInstructions(&BB);

    for (Instruction &I : BB) {
      LLVM_DEBUG(dbgs() << "\nInstruction: '" << I << "'\n");

      if (const auto *DL =
              cast_or_null<DILocation>(I.getMetadata(LLVMContext::MD_dbg))) {
        LLVM_DEBUG({
          dbgs() << "  Location: ";
          DL->dump(TheModule);
        });
        getOrCreateAbstractScope(DL);
      }

      for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange()))
        HandleDbgVariable(&DVR);

      if (options().getPrintAnyLine())
        constructLine(Scope, SP, I, GenerateLineBeforePrologue);

      InstrLineAddrMap[I.getIterator().getNodePtr()] = CurrentOffset;

      // Update code offset.
      updateLineOffset();
    }
    InstrLineAddrMap[BB.end().getNodePtr()] = CurrentOffset;
  }
  GenerateLineBeforePrologue = false;

  if (AddUnspecifiedParameters) {
    LVElement *Parameter = createElement(dwarf::DW_TAG_unspecified_parameters);
    if (Parameter) {
      Parameter->setIsFinalized();
      Scope->addElement(Parameter);
    }
  }

  LLVM_DEBUG({ dbgs() << "\nTraverse seen debug variables\n"; });
  for (const DebugVariableAggregate &DVA : SeenVars) {
    LLVM_DEBUG({ DbgValueRanges->printValues(DVA, dbgs()); });
    DILocalVariable *LV = const_cast<DILocalVariable *>(DVA.getVariable());
    LVSymbol *Symbol = getSymbolForSeenMD(LV);
    // Undefined only value, ignore.
    if (!Symbol)
      continue;

    DIType *Ty = LV->getType();
    uint64_t Size = Ty ? Ty->getSizeInBits() / CHAR_BIT : 1;
    LLVM_DEBUG({
      LV->dump(TheModule);
      Ty->dump(TheModule);
      dbgs() << "Type size: " << Size << "\n";
    });

    auto AddLocationOp = [&](Value *V, bool IsMem) {
      uint64_t RegValue = ValueNameMap.addValue(V);
      if (IsMem)
        Symbol->addLocationOperands(dwarf::DW_OP_bregx, {RegValue, 0});
      else
        Symbol->addLocationOperands(dwarf::DW_OP_regx, RegValue);
    };

    auto AddLocation = [&](DbgValueDef DV) {
      bool IsMem = DV.IsMemory;
      DIExpression *CanonicalExpr = const_cast<DIExpression *>(
          DIExpression::convertToVariadicExpression(DV.Expression));
      RawLocationWrapper Locations(DV.Locations);
      for (DIExpression::ExprOperand ExprOp : CanonicalExpr->expr_ops()) {
        if (ExprOp.getOp() == dwarf::DW_OP_LLVM_arg) {
          AddLocationOp(Locations.getVariableLocationOp(ExprOp.getArg(0)),
                        IsMem);
        } else {
          if (ExprOp.getOp() > std::numeric_limits<uint8_t>::max())
            LLVM_DEBUG(dbgs() << "Bad DWARF op: " << ExprOp.getOp() << "\n");
          uint8_t ShortOp = (uint8_t)ExprOp.getOp();
          Symbol->addLocationOperands(
              ShortOp,
              ArrayRef<uint64_t>(std::next(ExprOp.get()), ExprOp.getNumArgs()));
        }
      }
    };

    if (DbgValueRanges->hasSingleLocEntry(DVA)) {
      DbgValueDef DV = DbgValueRanges->getSingleLoc(DVA);
      Symbol->addLocation(llvm::dwarf::DW_AT_location, /*LowPC=*/0,
                          /*HighPC=*/-1, /*SectionOffset=*/0,
                          /*OffsetOnEntry=*/0);
      assert(DV.IsMemory && "Single location should be memory!");
      AddLocation(DV);
    } else {
      for (const DbgRangeEntry &Entry :
           DbgValueRanges->getVariableRanges(DVA)) {
        // These line addresses should have already been inserted into the
        // InstrLineAddrMap, so we assume they are present here.
        LVOffset Start = InstrLineAddrMap.at(Entry.Start.getNodePtr());
        LVOffset End = InstrLineAddrMap.at(Entry.End.getNodePtr());
        Symbol->addLocation(llvm::dwarf::DW_AT_location, Start, End,
                            /*SectionOffset=*/0, /*OffsetOnEntry=*/0);
        DbgValueDef DV = Entry.Value;
        AddLocation(DV);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// IR Reader entry point.
//===----------------------------------------------------------------------===//
Error LVIRReader::createScopes() {
  LLVM_DEBUG({
    W.startLine() << "\n";
    W.printString("File", getFilename());
    W.printString("Format", FileFormatName);
  });

  // The IR Reader supports only debug records.
  // We identify the debug input format and if it is intrinsics, it is
  // converted to the debug records.
  if (Error Err = LVReader::createScopes())
    return Err;

  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIR(
      BitCodeIR ? BitCodeIR->getMemoryBufferRef() : *TextualIR, Err, Context);
  if (!M)
    return createStringError(errc::invalid_argument,
                             "Could not create IR module for: %s",
                             getFilename().str().c_str());

  TheModule = M.get();
  if (!TheModule->getNamedMetadata("llvm.dbg.cu")) {
    LLVM_DEBUG(dbgs() << "Skipping module without debug info\n");
    return Error::success();
  }

  DwarfVersion = TheModule->getDwarfVersion();

  LLVM_DEBUG({ dbgs() << "\nProcess CompileUnits\n"; });
  for (const DICompileUnit *CU : TheModule->debug_compile_units()) {
    LLVM_DEBUG({
      dbgs() << "\nCU: ";
      CU->dump(TheModule);
    });

    CompileUnit = static_cast<LVScopeCompileUnit *>(constructElement(CU));
    CUNode = const_cast<DICompileUnit *>(CU);

    const DIFile *File = CU->getFile();
    CompileUnit->setName(File->getFilename());
    CompileUnit->setCompilationDirectory(File->getDirectory());
    CompileUnit->setIsFinalized();

    Root->addElement(CompileUnit);

    uint16_t LanguageName = CU->getSourceLanguage().getUnversionedName();
    LVSourceLanguage SL =
        TheModule->getCodeViewFlag()
            ? LVSourceLanguage(
                  static_cast<llvm::codeview::SourceLanguage>(LanguageName))
            : LVSourceLanguage(
                  static_cast<llvm::dwarf::SourceLanguage>(LanguageName));
    setDefaultLowerBound(&SL);

    if (options().getAttributeLanguage())
      CompileUnit->setSourceLanguage(SL);

    if (options().getAttributeProducer())
      CompileUnit->setProducer(CU->getProducer());

    // Global Variables.
    LLVM_DEBUG({ dbgs() << "\nGlobal Variables\n"; });
    for (const DIGlobalVariableExpression *GVE : CU->getGlobalVariables())
      getOrCreateVariable(GVE);

    // The enumeration types need to be created, regardless if they are
    // nested to any other aggregate type, as they are not included in
    // their elements. But their scope is correct (aggregate).
    LLVM_DEBUG({ dbgs() << "\nEnumeration Types\n"; });
    for (auto *ET : CU->getEnumTypes())
      getOrCreateType(ET);

    // Retained types.
    LLVM_DEBUG({ dbgs() << "\nRetained Types\n"; });
    for (const auto *RT : CU->getRetainedTypes()) {
      if (const auto *Ty = dyn_cast<DIType>(RT)) {
        getOrCreateType(Ty);
      } else {
        getOrCreateSubprogram(cast<DISubprogram>(RT));
      }
    }

    // Imported entities.
    LLVM_DEBUG({ dbgs() << "\nImported Entities\n"; });
    for (const auto *IE : CU->getImportedEntities())
      constructImportedEntity(CompileUnit, IE);
  }

  // Traverse Functions.
  LLVM_DEBUG({
    dbgs() << "\nFunctions\n";
    for (Function &F : M->getFunctionList())
      if (const auto *SP = cast_or_null<DISubprogram>(F.getSubprogram()))
        SP->dump(TheModule);
  });

  for (Function &F : M->getFunctionList())
    processBasicBlocks(F);

  // Perform extra tasks on the created scopes.
  resolveInlinedLexicalScopes();
  removeEmptyScopes();

  processLocationGaps();
  processScopes();

  if (options().getInternalIntegrity())
    checkScopes(CompileUnit);

  TheModule = nullptr;
  return Error::success();
}

void LVIRReader::constructRange(LVScope *Scope, LVAddress LowPC,
                                LVAddress HighPC) {
  assert(Scope && "Invalid logical element");
  LLVM_DEBUG({
    dbgs() << "\n[constructRange]\n";
    dbgs() << "ID: " << hexString(Scope->getID()) << " ";
    dbgs() << "LowPC: " << hexString(LowPC) << " ";
    dbgs() << "HighPC: " << hexString(HighPC) << " ";
    dbgs() << "Name: " << Scope->getName() << "\n";
  });

  // Process ranges base on logical lines.
  Scope->addObject(LowPC, HighPC);
  if (!Scope->getIsCompileUnit()) {
    // If the scope is a function, add it to the public names.
    if ((options().getAttributePublics() || options().getPrintAnyLine()) &&
        Scope->getIsFunction() && !Scope->getIsInlinedFunction())
      CompileUnit->addPublicName(Scope, LowPC, HighPC);
  }
  addSectionRange(/*SectionIndex=*/0, Scope, LowPC, HighPC);

  // Replicate DWARF reader funtionality of processing DW_AT_ranges for
  // the compilation unit.
  CompileUnit->addObject(LowPC, HighPC);
  addSectionRange(/*SectionIndex=*/0, CompileUnit, LowPC, HighPC);
}

// Create the location ranges for the given scope and in the case of
// functions, generate an entry in the public names set.
void LVIRReader::constructRange(LVScope *Scope) {
  LLVM_DEBUG({
    dbgs() << "\n[constructRange]\n";
    dbgs() << "ID: " << hexString(Scope->getID()) << " ";
    dbgs() << "Name: " << Scope->getName() << "\n\n";
  });

  auto NextRange = [&](LVAddress Offset) -> LVAddress {
    return Offset + OffsetIncrease - 1;
  };

  // Get any logical lines.
  const LVLines *Lines = Scope->getLines();
  if (!Lines)
    return;

  // Traverse the logical lines and build the logical ranges.
  LVAddress Lower = 0;
  LVAddress Upper = 0;
  LVAddress Current = 0;
  LVAddress Previous = 0;
  for (const LVLine *Line : *Lines) {
    LLVM_DEBUG({
      dbgs() << "[" << hexString(Line->getAddress()) << "] ";
      dbgs() << "LineNo: " << decString(Line->getLineNumber()) << "\n";
      dbgs() << "Lower: " << hexString(Lower) << " ";
      dbgs() << "Upper: " << hexString(Upper) << " ";
      dbgs() << "Previous: " << hexString(Previous) << " ";
      dbgs() << "Current: " << hexString(Current) << "\n";
    });
    if (!Upper) {
      // First line in range.
      Lower = Line->getAddress();
      Upper = NextRange(Lower);
      Current = Lower;
      continue;
    }
    Previous = Current;
    Current = Line->getAddress();
    if (Current == Previous) {
      // Contiguous lines at the same address (Debug and its assembler).
      continue;
    }
    if (Current == Upper + 1) {
      // There is no gap.
      Upper = NextRange(Current);
    } else {
      // There is a gap.
      constructRange(Scope, Lower, Upper);
      Lower = Current;
      Upper = NextRange(Lower);
    }
  }
  constructRange(Scope, Lower, Upper);
}

// At this point, all scopes for the compile unit has been created.
// The following aditional steps need to be performed on them:
// - If the lexical block doesn't have non-scope children, skip its
//   emission and put its children directly to the parent scope.
// The '--internal=id' is turned on just for debugging traces. Then
// it is turned to its previous state.
void LVIRReader::removeEmptyScopes() {
  LLVM_DEBUG({ dbgs() << "\n[removeEmptyScopes]\n"; });

  SmallVector<LVScope *> EmptyScopes;

  // Delete lexically empty scopes.
  auto DeleteEmptyScopes = [&]() {
    if (EmptyScopes.empty())
      return;

    LLVM_DEBUG({
      dbgs() << "\n** Collected empty scopes **\n";
      for (auto Scope : EmptyScopes)
        Scope->print(dbgs());
    });

    LVScope *Parent = nullptr;
    for (auto Scope : EmptyScopes) {
      Parent = Scope->getParentScope();
      LLVM_DEBUG({
        dbgs() << "Scope: " << Scope->getID() << ", ";
        dbgs() << "Parent: " << Parent->getID() << "\n";
      });

      // If the target scope has lines, move them to the scope parent.
      const LVLines *Lines = Scope->getLines();
      if (Lines) {
        LVLines Pack;
        std::copy(Lines->begin(), Lines->end(), std::back_inserter(Pack));
        for (LVLine *Line : Pack) {
          if (Scope->removeElement(Line)) {
            LLVM_DEBUG({ dbgs() << "Line: " << Line->getID() << "\n"; });
            Line->resetParent();
            Parent->addElement(Line);
            Line->updateLevel(Parent, /*Moved=*/false);
          }
        }
      }

      if (Parent->removeElement(Scope)) {
        const LVScopes *Scopes = Scope->getScopes();
        if (Scopes) {
          for (LVScope *Child : *Scopes) {
            LLVM_DEBUG({ dbgs() << "Child: " << Child->getID() << "\n"; });
            Child->resetParent();
            Parent->addElement(Child);
            Child->updateLevel(Parent, /*Moved=*/false);
          }
        }
      }
    }
  };

  // Traverse the scopes tree and collect those lexical blocks that does not
  // have non-scope children. Do not include the lines as they are included
  // in the logical view as a way to show their associated logical scope.
  std::function<void(LVScope *)> TraverseScope = [&](LVScope *Current) {
    auto IsEmpty = [](LVScope *Scope) -> bool {
      return !Scope->getSymbols() && !Scope->getTypes() && !Scope->getRanges();
    };

    if (const LVScopes *Scopes = Current->getScopes()) {
      for (LVScope *Scope : *Scopes) {
        if (Scope->getIsLexicalBlock() && IsEmpty(Scope))
          EmptyScopes.push_back(Scope);
        TraverseScope(Scope);
      }
    }
  };

  // Preserve current setting for '--internal=id'.
  bool InternalID = options().getInternalID();
  options().setInternalID();

  LLVM_DEBUG({
    dbgs() << "\nBefore - RemoveEmptyScopes\n";
    printCollectedElements(Root);
  });

  TraverseScope(CompileUnit);
  DeleteEmptyScopes();

  LLVM_DEBUG({
    dbgs() << "\nAfter - RemoveEmptyScopes\n";
    printCollectedElements(Root);
  });

  // Restore setting for '--internal=id'.
  if (!InternalID)
    options().resetInternalID();
}

// The IR generated by Clang, allocates the inlined lexical scopes
// at the enclosing function level. Move them to the correct scope.
void LVIRReader::resolveInlinedLexicalScopes() {
  LLVM_DEBUG({ dbgs() << "\n[resolveInlinedLexicalScopes]\n"; });
  LLVM_DEBUG({ dumpInlinedInfo("Before", /*Full=*/false); });

  std::function<void(LVScope * Scope)> TraverseChildren = [&](LVScope *Parent) {
    LLVM_DEBUG({
      dbgs() << "\nParent Scope: ";
      Parent->dumpCommon();
    });

    // Get associated inlined scopes for the parent scope.
    LVList &ParentInlinedList = getInlinedList(Parent);

    // Check if the inlined scope parent is in the ParentInlinedList.
    auto CheckInlinedScope = [&](LVList &ScopeInlinedList) -> bool {
      bool Matched = true;
      for (auto &InlinedScope : ScopeInlinedList) {
        LLVM_DEBUG({
          dbgs() << "Inlined Scope:  ";
          InlinedScope->dumpCommon();
        });
        LVScope *ParentScope = InlinedScope->getParentScope();
        for (auto &ParentInlinedScope : ParentInlinedList) {
          if (ParentInlinedScope != ParentScope) {
            // If the parent for the inlined scope is not the Parent Inlined
            // list, it means the lexical scope is incorrect.
            // Stop the traversal as the other inlined scopes will have the
            // same problem as they were created from the same original scope.
            LLVM_DEBUG({
              dbgs() << "\nIncorrect parent scope\n";
              dbgs() << "ParentInlinedScope: ";
              ParentInlinedScope->dumpCommon();
              dbgs() << "ParentScope: ";
              ParentScope->dumpCommon();
              dbgs() << "\n";
            });
            Matched = false;
            break;
          }
        }
        if (!Matched)
          break;
      }
      return Matched;
    };

    // Adjust the inlined scopes based on the ParentInlinedList.
    auto AdjustInlinedScope = [&](LVList &ScopeInlinedList) {
      assert(ScopeInlinedList.size() == ParentInlinedList.size() &&
             "Scope list do not have same number of items.");

      LLVM_DEBUG({ dbgs() << "Begin scope adjustment\n"; });
      LVScope *CurrentParent = nullptr;
      LVScope *TargetParent = nullptr;
      LVScope *InlinedScope = nullptr;
      auto ItInlined = ScopeInlinedList.begin();
      auto ItParent = ParentInlinedList.begin();
      while (ItInlined != ScopeInlinedList.end()) {
        TargetParent = *ItParent;
        InlinedScope = *ItInlined;
        CurrentParent = InlinedScope->getParentScope();

        LLVM_DEBUG({
          dbgs() << "Target Parent:  ";
          TargetParent->dumpCommon();
          dbgs() << "Current Parent: ";
          CurrentParent->dumpCommon();
          dbgs() << "Inlined:        ";
          InlinedScope->dumpCommon();
        });

        // Correct lexical scope.
        if (CurrentParent->removeElement(InlinedScope)) {
          TargetParent->addElement(InlinedScope);
          InlinedScope->updateLevel(TargetParent, /*Moved=*/false);
        }
        ++ItInlined;
        ++ItParent;
      }
      LLVM_DEBUG({ dbgs() << "End scope adjustment\n"; });
    };

    // Traverse the scope children.
    if (const LVScopes *Children = Parent->getScopes())
      for (LVScope *Scope : *Children) {
        LLVM_DEBUG({
          dbgs() << "\nOrigin Scope: ";
          Scope->dumpCommon();
        });

        // Get associated inlined scopes for the scope.
        LVList &ScopeInlinedList = getInlinedList(Scope);
        if (!CheckInlinedScope(ScopeInlinedList)) {
          // AdjustInlinedScope to the correct lexical scope.
          AdjustInlinedScope(ScopeInlinedList);
        }
        TraverseChildren(Scope);
      }
  };

  // Traverse the origin scopes and for each function scope, analyze their
  // associated inlined scopes to see if they have to be move to their
  // correct lexical scope.
  for (auto &Entry : InlinedList) {
    LVScope *OriginScope = Entry.first;
    if (OriginScope->getIsFunction())
      TraverseChildren(OriginScope);
  }

  LLVM_DEBUG({ dumpInlinedInfo("After", /*Full=*/false); });
}

// During the IR-to-logical-view construction, traverse all the logical
// elements to check if they have been properly constructed (finalized).
void LVIRReader::checkScopes(LVScope *Scope) {
  LLVM_DEBUG({ dbgs() << "\n[checkScopes]\n"; });

  auto PrintElement = [](LVElement *Element) {
    LLVM_DEBUG({
      dwarf::Tag Tag = Element->getTag();
      size_t ID = Element->getID();
      const char *Kind = Element->kind();
      StringRef Name = Element->getName();
      uint32_t LineNumber = Element->getLineNumber();
      dbgs() << "Tag: "
             << formatv("{0} ", fmt_align(Tag, AlignStyle::Left, 35));
      dbgs() << "ID: " << formatv("{0} ", fmt_align(ID, AlignStyle::Left, 5));
      dbgs() << "Kind: "
             << formatv("{0} ", fmt_align(Kind, AlignStyle::Left, 15));
      dbgs() << "Line: "
             << formatv("{0} ", fmt_align(LineNumber, AlignStyle::Left, 5));
      dbgs() << "Name: '" << std::string(Name) << "' ";
      dbgs() << "\n";
    });
  };

  std::function<void(LVScope * Parent)> Traverse = [&](LVScope *Current) {
    auto Check = [&](auto *Entry) {
      if (Entry)
        if (!Entry->getIsFinalized())
          PrintElement(Entry);
    };

    for (LVElement *Element : Current->getChildren())
      Check(Element);

    if (Current->getScopes())
      for (LVScope *Scope : *Current->getScopes())
        Traverse(Scope);
  };

  // Start traversing the scopes root and check its integrity.
  Traverse(Scope);
}

void LVIRReader::sortScopes() { Root->sort(); }

void LVIRReader::print(raw_ostream &OS) const {
  OS << "LVIRReader\n";
  LLVM_DEBUG(dbgs() << "CreateReaders\n");
}
