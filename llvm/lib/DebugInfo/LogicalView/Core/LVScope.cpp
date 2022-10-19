//===-- LVScope.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the LVScope class.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVScope.h"
#include "llvm/DebugInfo/LogicalView/Core/LVLine.h"
#include "llvm/DebugInfo/LogicalView/Core/LVOptions.h"
#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"
#include "llvm/DebugInfo/LogicalView/Core/LVSymbol.h"
#include "llvm/DebugInfo/LogicalView/Core/LVType.h"

using namespace llvm;
using namespace llvm::logicalview;

#define DEBUG_TYPE "Scope"

namespace {
const char *const KindArray = "Array";
const char *const KindBlock = "Block";
const char *const KindCallSite = "CallSite";
const char *const KindClass = "Class";
const char *const KindCompileUnit = "CompileUnit";
const char *const KindEnumeration = "Enumeration";
const char *const KindFile = "File";
const char *const KindFunction = "Function";
const char *const KindInlinedFunction = "InlinedFunction";
const char *const KindNamespace = "Namespace";
const char *const KindStruct = "Struct";
const char *const KindTemplateAlias = "TemplateAlias";
const char *const KindTemplatePack = "TemplatePack";
const char *const KindUndefined = "Undefined";
const char *const KindUnion = "Union";
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// DWARF lexical block, such as: namespace, function, compile unit, module, etc.
//===----------------------------------------------------------------------===//
LVScope::~LVScope() {
  delete Types;
  delete Symbols;
  delete Scopes;
  delete Lines;
  delete Children;
}

// Return a string representation for the scope kind.
const char *LVScope::kind() const {
  const char *Kind = KindUndefined;
  if (getIsArray())
    Kind = KindArray;
  else if (getIsBlock())
    Kind = KindBlock;
  else if (getIsCallSite())
    Kind = KindCallSite;
  else if (getIsCompileUnit())
    Kind = KindCompileUnit;
  else if (getIsEnumeration())
    Kind = KindEnumeration;
  else if (getIsInlinedFunction())
    Kind = KindInlinedFunction;
  else if (getIsNamespace())
    Kind = KindNamespace;
  else if (getIsTemplatePack())
    Kind = KindTemplatePack;
  else if (getIsRoot())
    Kind = KindFile;
  else if (getIsTemplateAlias())
    Kind = KindTemplateAlias;
  else if (getIsClass())
    Kind = KindClass;
  else if (getIsFunction())
    Kind = KindFunction;
  else if (getIsStructure())
    Kind = KindStruct;
  else if (getIsUnion())
    Kind = KindUnion;
  return Kind;
}

void LVScope::addToChildren(LVElement *Element) {
  if (!Children)
    Children = new LVElements();
  Children->push_back(Element);
}

void LVScope::addElement(LVElement *Element) {
  assert(Element && "Invalid element.");
  if (Element->getIsType())
    addElement(static_cast<LVType *>(Element));
  else if (Element->getIsScope())
    addElement(static_cast<LVScope *>(Element));
  else if (Element->getIsSymbol())
    addElement(static_cast<LVSymbol *>(Element));
  else if (Element->getIsLine())
    addElement(static_cast<LVLine *>(Element));
  else
    llvm_unreachable("Invalid Element.");
}

// Adds the line info item to the ones stored in the scope.
void LVScope::addElement(LVLine *Line) {
  assert(Line && "Invalid line.");
  assert(!Line->getParent() && "Line already inserted");
  if (!Lines)
    Lines = new LVAutoLines();

  // Add it to parent.
  Lines->push_back(Line);
  Line->setParent(this);

  // Notify the reader about the new element being added.
  getReaderCompileUnit()->addedElement(Line);

  // All logical elements added to the children, are sorted by any of the
  // following criterias: offset, name, line number, kind.
  // Do not add the line records to the children, as they represent the
  // logical view for the text section and any sorting will not preserve
  // the original sequence.

  // Indicate that this tree branch has lines.
  traverseParents(&LVScope::getHasLines, &LVScope::setHasLines);
}

// Adds the scope to the child scopes and sets the parent in the child.
void LVScope::addElement(LVScope *Scope) {
  assert(Scope && "Invalid scope.");
  assert(!Scope->getParent() && "Scope already inserted");
  if (!Scopes)
    Scopes = new LVAutoScopes();

  // Add it to parent.
  Scopes->push_back(Scope);
  addToChildren(Scope);
  Scope->setParent(this);

  // Notify the reader about the new element being added.
  getReaderCompileUnit()->addedElement(Scope);

  // If the element is a global reference, mark its parent as having global
  // references; that information is used, to print only those branches
  // with global references.
  if (Scope->getIsGlobalReference())
    traverseParents(&LVScope::getHasGlobals, &LVScope::setHasGlobals);
  else
    traverseParents(&LVScope::getHasLocals, &LVScope::setHasLocals);

  // Indicate that this tree branch has scopes.
  traverseParents(&LVScope::getHasScopes, &LVScope::setHasScopes);
}

// Adds a symbol to the ones stored in the scope.
void LVScope::addElement(LVSymbol *Symbol) {
  assert(Symbol && "Invalid symbol.");
  assert(!Symbol->getParent() && "Symbol already inserted");
  if (!Symbols)
    Symbols = new LVAutoSymbols();

  // Add it to parent.
  Symbols->push_back(Symbol);
  addToChildren(Symbol);
  Symbol->setParent(this);

  // Notify the reader about the new element being added.
  getReaderCompileUnit()->addedElement(Symbol);

  // If the element is a global reference, mark its parent as having global
  // references; that information is used, to print only those branches
  // with global references.
  if (Symbol->getIsGlobalReference())
    traverseParents(&LVScope::getHasGlobals, &LVScope::setHasGlobals);
  else
    traverseParents(&LVScope::getHasLocals, &LVScope::setHasLocals);

  // Indicate that this tree branch has symbols.
  traverseParents(&LVScope::getHasSymbols, &LVScope::setHasSymbols);
}

// Adds a type to the ones stored in the scope.
void LVScope::addElement(LVType *Type) {
  assert(Type && "Invalid type.");
  assert(!Type->getParent() && "Type already inserted");
  if (!Types)
    Types = new LVAutoTypes();

  // Add it to parent.
  Types->push_back(Type);
  addToChildren(Type);
  Type->setParent(this);

  // Notify the reader about the new element being added.
  getReaderCompileUnit()->addedElement(Type);

  // If the element is a global reference, mark its parent as having global
  // references; that information is used, to print only those branches
  // with global references.
  if (Type->getIsGlobalReference())
    traverseParents(&LVScope::getHasGlobals, &LVScope::setHasGlobals);
  else
    traverseParents(&LVScope::getHasLocals, &LVScope::setHasLocals);

  // Indicate that this tree branch has types.
  traverseParents(&LVScope::getHasTypes, &LVScope::setHasTypes);
}

bool LVScope::removeElement(LVElement *Element) {
  auto Predicate = [Element](LVElement *Item) -> bool {
    return Item == Element;
  };
  auto RemoveElement = [Element, Predicate](auto &Container) -> bool {
    auto Iter = std::remove_if(Container->begin(), Container->end(), Predicate);
    if (Iter != Container->end()) {
      Container->erase(Iter, Container->end());
      Element->resetParent();
      return true;
    }
    return false;
  };

  // As 'children' contains only (scopes, symbols and types), check if the
  // element we are deleting is a line.
  if (Element->getIsLine())
    return RemoveElement(Lines);

  if (RemoveElement(Children)) {
    if (Element->getIsSymbol())
      return RemoveElement(Symbols);
    if (Element->getIsType())
      return RemoveElement(Types);
    if (Element->getIsScope())
      return RemoveElement(Scopes);
    llvm_unreachable("Invalid element.");
  }

  return false;
}

void LVScope::addMissingElements(LVScope *Reference) {
  setAddedMissing();
  if (!Reference)
    return;

  // Get abstract symbols for the given scope reference.
  const LVSymbols *ReferenceSymbols = Reference->getSymbols();
  if (!ReferenceSymbols)
    return;

  LVSymbols References;
  References.append(ReferenceSymbols->begin(), ReferenceSymbols->end());

  auto RemoveSymbol = [&](LVSymbols &Symbols, LVSymbol *Symbol) {
    LVSymbols::iterator Iter = std::remove_if(
        Symbols.begin(), Symbols.end(),
        [Symbol](LVSymbol *Item) -> bool { return Item == Symbol; });
    if (Iter != Symbols.end())
      Symbols.erase(Iter, Symbols.end());
  };

  // Erase abstract symbols already in this scope from the collection of
  // symbols in the referenced scope.
  if (getSymbols())
    for (const LVSymbol *Symbol : *getSymbols())
      if (Symbol->getHasReferenceAbstract())
        RemoveSymbol(References, Symbol->getReference());

  // If we have elements left in 'References', those are the elements that
  // need to be inserted in the current scope.
  if (References.size()) {
    LLVM_DEBUG({
      dbgs() << "Insert Missing Inlined Elements\n"
             << "Offset = " << hexSquareString(getOffset()) << " "
             << "Abstract = " << hexSquareString(Reference->getOffset())
             << "\n";
    });
    for (LVSymbol *Reference : References) {
      LLVM_DEBUG({
        dbgs() << "Missing Offset = " << hexSquareString(Reference->getOffset())
               << "\n";
      });
      // We can't clone the abstract origin reference, as it contain extra
      // information that is incorrect for the element to be inserted.
      // As the symbol being added does not exist in the debug section,
      // use its parent scope offset, to indicate its DIE location.
      LVSymbol *Symbol = new LVSymbol();
      addElement(Symbol);
      Symbol->setOffset(getOffset());
      Symbol->setIsOptimized();
      Symbol->setReference(Reference);

      // The symbol can be a constant, parameter or variable.
      if (Reference->getIsConstant())
        Symbol->setIsConstant();
      else if (Reference->getIsParameter())
        Symbol->setIsParameter();
      else if (Reference->getIsVariable())
        Symbol->setIsVariable();
      else
        llvm_unreachable("Invalid symbol kind.");
    }
  }
}

void LVScope::updateLevel(LVScope *Parent, bool Moved) {
  // Update the level for the element itself and all its children, using the
  // given scope parent as reference.
  setLevel(Parent->getLevel() + 1);

  // Update the children.
  if (Children)
    for (LVElement *Element : *Children)
      Element->updateLevel(this, Moved);

  // Update any lines.
  if (Lines)
    for (LVLine *Line : *Lines)
      Line->updateLevel(this, Moved);
}

void LVScope::resolve() {
  if (getIsResolved())
    return;

  // Resolve the element itself.
  LVElement::resolve();

  // Resolve the children.
  if (Children)
    for (LVElement *Element : *Children) {
      if (getIsGlobalReference())
        // If the scope is a global reference, mark all its children as well.
        Element->setIsGlobalReference();
      Element->resolve();
    }
}

void LVScope::resolveName() {
  if (getIsResolvedName())
    return;
  setIsResolvedName();

  // If the scope is a template, resolve the template parameters and get
  // the name for the template with the encoded arguments.
  if (getIsTemplate())
    resolveTemplate();
  else {
    if (LVElement *BaseType = getType()) {
      BaseType->resolveName();
      resolveFullname(BaseType);
    }
  }

  // In the case of unnamed scopes, try to generate a name for it, using
  // the parents name and the line information. In the case of compiler
  // generated functions, use its linkage name if is available.
  if (!isNamed()) {
    if (getIsArtificial())
      setName(getLinkageName());
    else
      generateName();
  }

  LVElement::resolveName();
}

void LVScope::resolveReferences() {
  // The scopes can have the following references to other elements:
  //   A type:
  //     DW_AT_type             ->  Type or Scope
  //     DW_AT_import           ->  Type
  //   A Reference:
  //     DW_AT_specification    ->  Scope
  //     DW_AT_abstract_origin  ->  Scope
  //     DW_AT_extension        ->  Scope

  // Resolve any referenced scope.
  LVScope *Reference = getReference();
  if (Reference) {
    Reference->resolve();
    // Recursively resolve the scope names.
    resolveReferencesChain();
  }

  // Set the file/line information using the Debug Information entry.
  setFile(Reference);

  // Resolve any referenced type or scope.
  if (LVElement *Element = getType())
    Element->resolve();
}

void LVScope::resolveElements() {
  // The current element represents the Root. Traverse each Compile Unit.
  if (!Scopes)
    return;

  for (LVScope *Scope : *Scopes) {
    LVScopeCompileUnit *CompileUnit = static_cast<LVScopeCompileUnit *>(Scope);
    getReader().setCompileUnit(CompileUnit);
    CompileUnit->resolve();
  }
}

StringRef LVScope::resolveReferencesChain() {
  // If the scope has a DW_AT_specification or DW_AT_abstract_origin,
  // follow the chain to resolve the name from those references.
  if (getHasReference() && !isNamed())
    setName(getReference()->resolveReferencesChain());

  return getName();
}

// Get template parameter types.
bool LVScope::getTemplateParameterTypes(LVTypes &Params) {
  // Traverse the scope types and populate the given container with those
  // types that are template parameters; that container will be used by
  // 'encodeTemplateArguments' to resolve them.
  if (const LVTypes *Types = getTypes())
    for (LVType *Type : *Types)
      if (Type->getIsTemplateParam()) {
        Type->resolve();
        Params.push_back(Type);
      }

  return !Params.empty();
}

// Resolve the template parameters/arguments relationship.
void LVScope::resolveTemplate() {
  if (getIsTemplateResolved())
    return;
  setIsTemplateResolved();

  // Check if we need to encode the template arguments.
  if (options().getAttributeEncoded()) {
    LVTypes Params;
    if (getTemplateParameterTypes(Params)) {
      std::string EncodedArgs;
      // Encode the arguments as part of the template name and update the
      // template name, to reflect the encoded parameters.
      encodeTemplateArguments(EncodedArgs, &Params);
      setEncodedArgs(EncodedArgs.c_str());
    }
  }
}

// Get the qualified name for the template.
void LVScope::getQualifiedName(std::string &QualifiedName) const {
  if (getIsRoot() || getIsCompileUnit())
    return;

  if (LVScope *Parent = getParentScope())
    Parent->getQualifiedName(QualifiedName);
  if (!QualifiedName.empty())
    QualifiedName.append("::");
  QualifiedName.append(std::string(getName()));
}

// Encode the template arguments as part of the template name.
void LVScope::encodeTemplateArguments(std::string &Name) const {
  // Qualify only when we are expanding parameters that are template
  // instances; the debugger will assume the current scope symbol as
  // the qualifying tag for the symbol being generated, which gives:
  //   namespace std {
  //     ...
  //     set<float,std::less<float>,std::allocator<float>>
  //     ...
  //   }
  // The 'set' symbol is assumed to have the qualified tag 'std'.

  // We are resolving a template parameter which is another template. If
  // it is already resolved, just get the qualified name and return.
  std::string BaseName;
  getQualifiedName(BaseName);
  if (getIsTemplateResolved())
    Name.append(BaseName);
}

void LVScope::encodeTemplateArguments(std::string &Name,
                                      const LVTypes *Types) const {
  // The encoded string will start with the scope name.
  Name.append("<");

  // The list of types are the template parameters.
  if (Types) {
    bool AddComma = false;
    for (const LVType *Type : *Types) {
      if (AddComma)
        Name.append(", ");
      Type->encodeTemplateArgument(Name);
      AddComma = true;
    }
  }

  Name.append(">");
}

bool LVScope::resolvePrinting() const {
  bool Globals = options().getAttributeGlobal();
  bool Locals = options().getAttributeLocal();
  if ((Globals && Locals) || (!Globals && !Locals)) {
    // Print both Global and Local.
  } else {
    // Check for Global or Local Objects.
    if ((Globals && !(getHasGlobals() || getIsGlobalReference())) ||
        (Locals && !(getHasLocals() || !getIsGlobalReference())))
      return false;
  }

  // For the case of functions, skip it if is compiler generated.
  if (getIsFunction() && getIsArtificial() &&
      !options().getAttributeGenerated())
    return false;

  return true;
}

Error LVScope::doPrint(bool Split, bool Match, bool Print, raw_ostream &OS,
                       bool Full) const {
  // During a view output splitting, use the output stream created by the
  // split context, then switch to the reader output stream.
  raw_ostream *StreamSplit = &OS;

  // If 'Split', we use the scope name (CU name) as the ouput file; the
  // delimiters in the pathname, must be replaced by a normal character.
  if (getIsCompileUnit()) {
    getReader().setCompileUnit(const_cast<LVScope *>(this));
    if (Split) {
      std::string ScopeName(getName());
      if (std::error_code EC =
              getReaderSplitContext().open(ScopeName, ".txt", OS))
        return createStringError(EC, "Unable to create split output file %s",
                                 ScopeName.c_str());
      StreamSplit = static_cast<raw_ostream *>(&getReaderSplitContext().os());
    }
  }

  // Ignore discarded or stripped scopes (functions).
  bool DoPrint = (options().getAttributeDiscarded()) ? true : !getIsDiscarded();

  // If we are in compare mode, the only conditions are related to the
  // element being missing. In the case of elements comparison, we print the
  // augmented view, that includes added elements.
  // In print mode, we check other conditions, such as local, global, etc.
  if (DoPrint) {
    DoPrint =
        getIsInCompare() ? options().getReportExecute() : resolvePrinting();
  }

  // At this point we have checked for very specific options, to decide if the
  // element will be printed. Include the caller's test for element general
  // print.
  DoPrint = DoPrint && (Print || options().getOutputSplit());

  if (DoPrint) {
    // Print the element itself.
    print(*StreamSplit, Full);

    // Check if we have reached the requested lexical level specified in the
    // command line options. Input file is level zero and the CU is level 1.
    if ((getIsRoot() || options().getPrintAnyElement()) &&
        options().getPrintFormatting() &&
        getLevel() < options().getOutputLevel()) {
      // Print the children.
      if (Children)
        for (const LVElement *Element : *Children) {
          if (Match && !Element->getHasPattern())
            continue;
          if (Error Err =
                  Element->doPrint(Split, Match, Print, *StreamSplit, Full))
            return Err;
        }

      // Print the line records.
      if (Lines)
        for (const LVLine *Line : *Lines) {
          if (Match && !Line->getHasPattern())
            continue;
          if (Error Err =
                  Line->doPrint(Split, Match, Print, *StreamSplit, Full))
            return Err;
        }
    }
  }

  // Done printing the compile unit. Print any requested summary and
  // restore the original output context.
  if (getIsCompileUnit()) {
    if (options().getPrintSummary())
      printSummary(*StreamSplit);
    if (options().getPrintSizes())
      printSizes(*StreamSplit);
    if (Split) {
      getReaderSplitContext().close();
      StreamSplit = &getReader().outputStream();
    }
  }

  return Error::success();
}

void LVScope::sort() {
  // Preserve the lines order as they are associated with user code.
  LVSortFunction SortFunction = getSortFunction();
  if (SortFunction) {
    std::function<void(LVScope * Parent, LVSortFunction SortFunction)> Sort =
        [&](LVScope *Parent, LVSortFunction SortFunction) {
          auto Traverse = [&](auto *Set, LVSortFunction SortFunction) {
            if (Set)
              std::stable_sort(Set->begin(), Set->end(), SortFunction);
          };
          Traverse(Parent->Types, SortFunction);
          Traverse(Parent->Symbols, SortFunction);
          Traverse(Parent->Scopes, SortFunction);
          Traverse(Parent->Children, SortFunction);

          if (Parent->Scopes)
            for (LVScope *Scope : *Parent->Scopes)
              Sort(Scope, SortFunction);
        };

    // Start traversing the scopes root and transform the element name.
    Sort(this, SortFunction);
  }
}

void LVScope::traverseParents(LVScopeGetFunction GetFunction,
                              LVScopeSetFunction SetFunction) {
  // Traverse the parent tree.
  LVScope *Parent = this;
  while (Parent) {
    // Terminates if the 'SetFunction' has been already executed.
    if ((Parent->*GetFunction)())
      break;
    (Parent->*SetFunction)();
    Parent = Parent->getParentScope();
  }
}

void LVScope::traverseParentsAndChildren(LVObjectGetFunction GetFunction,
                                         LVObjectSetFunction SetFunction) {
  if (options().getReportParents()) {
    // First traverse the parent tree.
    LVScope *Parent = this;
    while (Parent) {
      // Terminates if the 'SetFunction' has been already executed.
      if ((Parent->*GetFunction)())
        break;
      (Parent->*SetFunction)();
      Parent = Parent->getParentScope();
    }
  }

  std::function<void(LVScope * Scope)> TraverseChildren = [&](LVScope *Scope) {
    auto Traverse = [&](const auto *Set) {
      if (Set)
        for (const auto &Entry : *Set)
          (Entry->*SetFunction)();
    };

    (Scope->*SetFunction)();

    Traverse(Scope->getTypes());
    Traverse(Scope->getSymbols());
    Traverse(Scope->getLines());

    if (const LVScopes *Scopes = Scope->getScopes())
      for (LVScope *Scope : *Scopes)
        TraverseChildren(Scope);
  };

  if (options().getReportChildren())
    TraverseChildren(this);
}

void LVScope::printEncodedArgs(raw_ostream &OS, bool Full) const {
  if (options().getPrintFormatting() && options().getAttributeEncoded())
    printAttributes(OS, Full, "{Encoded} ", const_cast<LVScope *>(this),
                    getEncodedArgs(), /*UseQuotes=*/false, /*PrintRef=*/false);
}

void LVScope::print(raw_ostream &OS, bool Full) const {
  if (getIncludeInPrint() && getReader().doPrintScope(this)) {
    // For a summary (printed elements), do not count the scope root.
    if (!(getIsRoot()))
      getReaderCompileUnit()->incrementPrintedScopes();
    LVElement::print(OS, Full);
    printExtra(OS, Full);
  }
}

void LVScope::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind());
  // Do not print any type or name for a lexical block.
  if (!getIsBlock()) {
    OS << " " << formattedName(getName());
    if (!getIsAggregate())
      OS << " -> " << typeOffsetAsString()
         << formattedNames(getTypeQualifiedName(), typeAsString());
  }
  OS << "\n";
}

//===----------------------------------------------------------------------===//
// DWARF Union/Structure/Class.
//===----------------------------------------------------------------------===//
void LVScopeAggregate::printExtra(raw_ostream &OS, bool Full) const {
  LVScope::printExtra(OS, Full);
  if (Full) {
    if (getIsTemplateResolved())
      printEncodedArgs(OS, Full);
    LVScope *Reference = getReference();
    if (Reference)
      Reference->printReference(OS, Full, const_cast<LVScopeAggregate *>(this));
  }
}

//===----------------------------------------------------------------------===//
// DWARF Template alias.
//===----------------------------------------------------------------------===//
void LVScopeAlias::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind()) << " " << formattedName(getName()) << " -> "
     << typeOffsetAsString()
     << formattedNames(getTypeQualifiedName(), typeAsString()) << "\n";
}

//===----------------------------------------------------------------------===//
// DWARF array (DW_TAG_array_type).
//===----------------------------------------------------------------------===//
void LVScopeArray::resolveExtra() {
  // If the scope is an array, resolve the subrange entries and get those
  // values encoded and assigned to the scope type.
  // Encode the array subrange entries as part of the name.
  if (getIsArrayResolved())
    return;
  setIsArrayResolved();

  // There are 2 cases to represent the bounds information for an array:
  // 1) DW_TAG_array_type
  //      DW_AT_type --> ref_type
  //      DW_TAG_subrange_type
  //        DW_AT_type --> ref_type (type of object)
  //        DW_AT_count --> value (number of elements in subrange)

  // 2) DW_TAG_array_type
  //      DW_AT_type --> ref_type
  //        DW_TAG_subrange_type
  //          DW_AT_lower_bound --> value
  //          DW_AT_upper_bound --> value

  // The idea is to represent the bounds as a string, depending on the format:
  // 1) [count]
  // 2) [lower][upper]

  // Traverse scope types, looking for those types that are subranges.
  LVTypes Subranges;
  if (const LVTypes *Types = getTypes())
    for (LVType *Type : *Types)
      if (Type->getIsSubrange()) {
        Type->resolve();
        Subranges.push_back(Type);
      }

  // Use the subrange types to generate the high level name for the array.
  // Check the type has been fully resolved.
  if (LVElement *BaseType = getType()) {
    BaseType->resolveName();
    resolveFullname(BaseType);
  }

  // In 'resolveFullname' a check is done for double spaces in the type name.
  std::stringstream ArrayInfo;
  if (ElementType)
    ArrayInfo << getTypeName().str() << " ";

  for (const LVType *Type : Subranges) {
    if (Type->getIsSubrangeCount())
      // Check if we have DW_AT_count subrange style.
      ArrayInfo << "[" << Type->getCount() << "]";
    else {
      // Get lower and upper subrange values.
      unsigned LowerBound;
      unsigned UpperBound;
      std::tie(LowerBound, UpperBound) = Type->getBounds();

      // The representation depends on the bound values. If the lower value
      // is zero, treat the pair as the elements count. Otherwise, just use
      // the pair, as they are representing arrays in languages other than
      // C/C++ and the lower limit is not zero.
      if (LowerBound)
        ArrayInfo << "[" << LowerBound << ".." << UpperBound << "]";
      else
        ArrayInfo << "[" << UpperBound + 1 << "]";
    }
  }

  // Update the scope name, to reflect the encoded subranges.
  setName(ArrayInfo.str());
}

void LVScopeArray::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind()) << " " << typeOffsetAsString()
     << formattedName(getName()) << "\n";
}

//===----------------------------------------------------------------------===//
// An object file (single or multiple CUs).
//===----------------------------------------------------------------------===//
void LVScopeCompileUnit::addSize(LVScope *Scope, LVOffset Lower,
                                 LVOffset Upper) {
  LLVM_DEBUG({
    dbgs() << format(
        "CU [0x%08x], Scope [0x%08x], Range [0x%08x:0x%08x], Size = %d\n",
        getOffset(), Scope->getOffset(), Lower, Upper, Upper - Lower);
  });

  // There is no need to check for a previous entry, as we are traversing the
  // debug information in sequential order.
  LVOffset Size = Upper - Lower;
  Sizes[Scope] = Size;
  if (this == Scope)
    // Record contribution size for the compilation unit.
    CUContributionSize = Size;
}

LVLine *LVScopeCompileUnit::lineLowerBound(LVAddress Address) const {
  LVAddressToLine::const_iterator Iter = AddressToLine.lower_bound(Address);
  return (Iter != AddressToLine.end()) ? Iter->second : nullptr;
}

LVLine *LVScopeCompileUnit::lineUpperBound(LVAddress Address) const {
  LVAddressToLine::const_iterator Iter = AddressToLine.upper_bound(Address);
  if (Iter != AddressToLine.begin())
    Iter = std::prev(Iter);
  return (Iter != AddressToLine.end()) ? Iter->second : nullptr;
}

StringRef LVScopeCompileUnit::getFilename(size_t Index) const {
  if (Index <= 0 || Index > Filenames.size())
    return StringRef();
  return getStringPool().getString(Filenames[Index - 1]);
}

void LVScopeCompileUnit::incrementPrintedLines() { ++Printed.Lines; }
void LVScopeCompileUnit::incrementPrintedScopes() { ++Printed.Scopes; }
void LVScopeCompileUnit::incrementPrintedSymbols() { ++Printed.Symbols; }
void LVScopeCompileUnit::incrementPrintedTypes() { ++Printed.Types; }

// Values are used by '--summary' option (allocated).
void LVScopeCompileUnit::increment(LVLine *Line) {
  if (Line->getIncludeInPrint())
    ++Allocated.Lines;
}
void LVScopeCompileUnit::increment(LVScope *Scope) {
  if (Scope->getIncludeInPrint())
    ++Allocated.Scopes;
}
void LVScopeCompileUnit::increment(LVSymbol *Symbol) {
  if (Symbol->getIncludeInPrint())
    ++Allocated.Symbols;
}
void LVScopeCompileUnit::increment(LVType *Type) {
  if (Type->getIncludeInPrint())
    ++Allocated.Types;
}

// A new element has been added to the scopes tree. Take the following steps:
// Increase the added element counters, for printing summary.
void LVScopeCompileUnit::addedElement(LVLine *Line) { increment(Line); }
void LVScopeCompileUnit::addedElement(LVScope *Scope) { increment(Scope); }
void LVScopeCompileUnit::addedElement(LVSymbol *Symbol) { increment(Symbol); }
void LVScopeCompileUnit::addedElement(LVType *Type) { increment(Type); }

void LVScopeCompileUnit::printLocalNames(raw_ostream &OS, bool Full) const {
  if (!options().getPrintFormatting())
    return;

  // Calculate an indentation value, to preserve a nice layout.
  size_t Indentation = options().indentationSize() +
                       lineNumberAsString().length() +
                       indentAsString(getLevel() + 1).length() + 3;

  enum class Option { Directory, File };
  auto PrintNames = [&](Option Action) {
    StringRef Kind = Action == Option::Directory ? "Directory" : "File";
    std::set<std::string> UniqueNames;
    for (size_t Index : Filenames) {
      // In the case of missing directory name in the .debug_line table,
      // the returned string has a leading '/'.
      StringRef Name = getStringPool().getString(Index);
      size_t Pos = Name.rfind('/');
      if (Pos != std::string::npos)
        Name = (Action == Option::File) ? Name.substr(Pos + 1)
                                        : Name.substr(0, Pos);
      // Collect only unique names.
      UniqueNames.insert(std::string(Name));
    }
    for (const std::string &Name : UniqueNames)
      OS << std::string(Indentation, ' ') << formattedKind(Kind) << " "
         << formattedName(Name) << "\n";
  };

  if (options().getAttributeDirectories())
    PrintNames(Option::Directory);
  if (options().getAttributeFiles())
    PrintNames(Option::File);
}

void LVScopeCompileUnit::printTotals(raw_ostream &OS) const {
  OS << "\nTotals by lexical level:\n";
  for (size_t Index = 1; Index <= MaxSeenLevel; ++Index)
    OS << format("[%03d]: %10d (%6.2f%%)\n", Index, Totals[Index].first,
                 Totals[Index].second);
}

void LVScopeCompileUnit::printScopeSize(const LVScope *Scope, raw_ostream &OS) {
  LVSizesMap::const_iterator Iter = Sizes.find(Scope);
  if (Iter != Sizes.end()) {
    LVOffset Size = Iter->second;
    assert(CUContributionSize && "Invalid CU contribution size.");
    // Get a percentage rounded to two decimal digits. This avoids
    // implementation-defined rounding inside printing functions.
    float Percentage =
        rint((float(Size) / CUContributionSize) * 100.0 * 100.0) / 100.0;
    OS << format("%10d (%6.2f%%) : ", Size, Percentage);
    Scope->print(OS);

    // Keep record of the total sizes at each lexical level.
    LVLevel Level = Scope->getLevel();
    if (Level > MaxSeenLevel)
      MaxSeenLevel = Level;
    if (Level >= Totals.size())
      Totals.resize(2 * Level);
    Totals[Level].first += Size;
    Totals[Level].second += Percentage;
  }
}

void LVScopeCompileUnit::printSizes(raw_ostream &OS) const {
  // Recursively print the contributions for each scope.
  std::function<void(const LVScope *Scope)> PrintScope =
      [&](const LVScope *Scope) {
        if (Scope->getLevel() < options().getOutputLevel()) {
          if (const LVScopes *Scopes = Scope->getScopes())
            for (const LVScope *Scope : *Scopes) {
              printScopeSize(Scope, OS);
              PrintScope(Scope);
            }
        }
      };

  bool PrintScopes = options().getPrintScopes();
  if (!PrintScopes)
    options().setPrintScopes();
  getReader().setCompileUnit(const_cast<LVScopeCompileUnit *>(this));

  OS << "\nScope Sizes:\n";
  options().resetPrintFormatting();
  options().setPrintOffset();

  // Print the scopes regardless if the user has requested any scopes
  // printing. Set the option just to allow printing the contributions.
  printScopeSize(this, OS);
  PrintScope(this);

  // Print total scope sizes by level.
  printTotals(OS);

  options().resetPrintOffset();
  options().setPrintFormatting();

  if (!PrintScopes)
    options().resetPrintScopes();
}

void LVScopeCompileUnit::printSummary(raw_ostream &OS) const {
  printSummary(OS, Printed, "Printed");
}

// Print summary details for the scopes tree.
void LVScopeCompileUnit::printSummary(raw_ostream &OS, const LVCounter &Counter,
                                      const char *Header) const {
  std::string Separator = std::string(29, '-');
  auto PrintSeparator = [&]() { OS << Separator << "\n"; };
  auto PrintHeadingRow = [&](const char *T, const char *U, const char *V) {
    OS << format("%-9s%9s  %9s\n", T, U, V);
  };
  auto PrintDataRow = [&](const char *T, unsigned U, unsigned V) {
    OS << format("%-9s%9d  %9d\n", T, U, V);
  };

  OS << "\n";
  PrintSeparator();
  PrintHeadingRow("Element", "Total", Header);
  PrintSeparator();
  PrintDataRow("Scopes", Allocated.Scopes, Counter.Scopes);
  PrintDataRow("Symbols", Allocated.Symbols, Counter.Symbols);
  PrintDataRow("Types", Allocated.Types, Counter.Types);
  PrintDataRow("Lines", Allocated.Lines, Counter.Lines);
  PrintSeparator();
  PrintDataRow(
      "Total",
      Allocated.Scopes + Allocated.Symbols + Allocated.Lines + Allocated.Types,
      Counter.Scopes + Counter.Symbols + Counter.Lines + Counter.Types);
}

void LVScopeCompileUnit::printMatchedElements(raw_ostream &OS,
                                              bool UseMatchedElements) {
  LVSortFunction SortFunction = getSortFunction();
  if (SortFunction)
    std::stable_sort(MatchedElements.begin(), MatchedElements.end(),
                     SortFunction);

  // Check the type of elements required to be printed. 'MatchedElements'
  // contains generic elements (lines, scopes, symbols, types). If we have a
  // request to print any generic element, then allow the normal printing.
  if (options().getPrintAnyElement()) {
    if (UseMatchedElements)
      OS << "\n";
    print(OS);

    if (UseMatchedElements) {
      // Print the details for the matched elements.
      for (const LVElement *Element : MatchedElements)
        Element->print(OS);
    } else {
      // Print the view for the matched scopes.
      for (const LVScope *Scope : MatchedScopes) {
        Scope->print(OS);
        if (const LVElements *Elements = Scope->getChildren())
          for (LVElement *Element : *Elements)
            Element->print(OS);
      }
    }

    // Print any requested summary.
    if (options().getPrintSummary()) {
      // In the case of '--report=details' the matched elements are
      // already counted; just proceed to print any requested summary.
      // Otherwise, count them and print the summary.
      if (!options().getReportList()) {
        for (LVElement *Element : MatchedElements) {
          if (!Element->getIncludeInPrint())
            continue;
          if (Element->getIsType())
            ++Found.Types;
          else if (Element->getIsSymbol())
            ++Found.Symbols;
          else if (Element->getIsScope())
            ++Found.Scopes;
          else if (Element->getIsLine())
            ++Found.Lines;
          else
            assert(Element && "Invalid element.");
        }
      }
      printSummary(OS, Found, "Printed");
    }
  }

  // Check if we have a request to print sizes for the matched elements
  // that are scopes.
  if (options().getPrintSizes()) {
    OS << "\n";
    print(OS);

    OS << "\nScope Sizes:\n";
    printScopeSize(this, OS);
    for (LVElement *Element : MatchedElements)
      if (Element->getIsScope())
        // Print sizes only for scopes.
        printScopeSize(static_cast<LVScope *>(Element), OS);

    printTotals(OS);
  }
}

void LVScopeCompileUnit::print(raw_ostream &OS, bool Full) const {
  // Reset counters for printed and found elements.
  const_cast<LVScopeCompileUnit *>(this)->Found.reset();
  const_cast<LVScopeCompileUnit *>(this)->Printed.reset();

  if (getReader().doPrintScope(this) && options().getPrintFormatting())
    OS << "\n";

  LVScope::print(OS, Full);
}

void LVScopeCompileUnit::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind()) << " '" << getName() << "'\n";
  if (options().getPrintFormatting() && options().getAttributeProducer())
    printAttributes(OS, Full, "{Producer} ",
                    const_cast<LVScopeCompileUnit *>(this), getProducer(),
                    /*UseQuotes=*/true,
                    /*PrintRef=*/false);

  // Reset file index, to allow its children to print the correct filename.
  options().resetFilenameIndex();

  // Print any files, directories, public names.
  if (Full) {
    printLocalNames(OS, Full);
  }
}

//===----------------------------------------------------------------------===//
// DWARF enumeration (DW_TAG_enumeration_type).
//===----------------------------------------------------------------------===//
void LVScopeEnumeration::printExtra(raw_ostream &OS, bool Full) const {
  // Print the full type name.
  OS << formattedKind(kind()) << " " << (getIsEnumClass() ? "class " : "")
     << formattedName(getName());
  if (getHasType())
    OS << " -> " << typeOffsetAsString()
       << formattedNames(getTypeQualifiedName(), typeAsString());
  OS << "\n";
}

//===----------------------------------------------------------------------===//
// DWARF formal parameter pack (DW_TAG_GNU_formal_parameter_pack).
//===----------------------------------------------------------------------===//
void LVScopeFormalPack::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind()) << " " << formattedName(getName()) << "\n";
}

//===----------------------------------------------------------------------===//
// DWARF function.
//===----------------------------------------------------------------------===//
void LVScopeFunction::resolveReferences() {
  // Before we resolve any references to other elements, check if we have
  // to insert missing elements, that have been stripped, which will help
  // the logical view comparison.
  if (options().getAttributeInserted() && getHasReferenceAbstract() &&
      !getAddedMissing()) {
    // Add missing elements at the function scope.
    addMissingElements(getReference());
    if (Scopes)
      for (LVScope *Scope : *Scopes)
        if (Scope->getHasReferenceAbstract() && !Scope->getAddedMissing())
          Scope->addMissingElements(Scope->getReference());
  }

  LVScope::resolveReferences();

  // The DWARF 'extern' attribute is generated at the class level.
  // 0000003f DW_TAG_class_type "CLASS"
  //   00000048 DW_TAG_subprogram "bar"
  //	            DW_AT_external DW_FORM_flag_present
  // 00000070 DW_TAG_subprogram "bar"
  //   DW_AT_specification DW_FORM_ref4 0x00000048
  // If there is a reference linking the declaration and definition, mark
  // the definition as extern, to facilitate the logical view comparison.
  if (getHasReferenceSpecification()) {
    LVScope *Reference = getReference();
    if (Reference && Reference->getIsExternal()) {
      Reference->resetIsExternal();
      setIsExternal();
    }
  }

  // Resolve the function associated type.
  if (!getType())
    if (LVScope *Reference = getReference())
      setType(Reference->getType());
}

void LVScopeFunction::setName(StringRef ObjectName) {
  LVScope::setName(ObjectName);
  // Check for system generated functions.
  getReader().isSystemEntry(this, ObjectName);
}

void LVScopeFunction::resolveExtra() {
  // Check if we need to encode the template arguments.
  if (getIsTemplate())
    resolveTemplate();
}

void LVScopeFunction::printExtra(raw_ostream &OS, bool Full) const {
  LVScope *Reference = getReference();

  // Inline attributes based on the reference element.
  uint32_t InlineCode =
      Reference ? Reference->getInlineCode() : getInlineCode();

  // Accessibility depends on the parent (class, structure).
  uint32_t AccessCode = 0;
  if (getIsMember())
    AccessCode = getParentScope()->getIsClass() ? dwarf::DW_ACCESS_private
                                                : dwarf::DW_ACCESS_public;

  std::string Attributes =
      getIsCallSite()
          ? ""
          : formatAttributes(externalString(), accessibilityString(AccessCode),
                             inlineCodeString(InlineCode), virtualityString());

  OS << formattedKind(kind()) << " " << Attributes << formattedName(getName())
     << discriminatorAsString() << " -> " << typeOffsetAsString()
     << formattedNames(getTypeQualifiedName(), typeAsString()) << "\n";

  // Print any active ranges.
  if (Full) {
    if (getIsTemplateResolved())
      printEncodedArgs(OS, Full);
    if (Reference)
      Reference->printReference(OS, Full, const_cast<LVScopeFunction *>(this));
  }
}

//===----------------------------------------------------------------------===//
// DWARF inlined function (DW_TAG_inlined_function).
//===----------------------------------------------------------------------===//
void LVScopeFunctionInlined::resolveExtra() {
  // Check if we need to encode the template arguments.
  if (getIsTemplate())
    resolveTemplate();
}

void LVScopeFunctionInlined::printExtra(raw_ostream &OS, bool Full) const {
  LVScopeFunction::printExtra(OS, Full);
}

//===----------------------------------------------------------------------===//
// DWARF subroutine type.
//===----------------------------------------------------------------------===//
// Resolve a Subroutine Type (Callback).
void LVScopeFunctionType::resolveExtra() {
  if (getIsMemberPointerResolved())
    return;
  setIsMemberPointerResolved();

  // The encoded string has the return type and the formal parameters type.
  std::string Name(typeAsString());
  Name.append(" (*)");
  Name.append("(");

  // Traverse the scope symbols, looking for those which are parameters.
  if (const LVSymbols *Symbols = getSymbols()) {
    bool AddComma = false;
    for (LVSymbol *Symbol : *Symbols)
      if (Symbol->getIsParameter()) {
        Symbol->resolve();
        if (LVElement *Type = Symbol->getType())
          Type->resolveName();
        if (AddComma)
          Name.append(", ");
        Name.append(std::string(Symbol->getTypeName()));
        AddComma = true;
      }
  }

  Name.append(")");

  // Update the scope name, to reflect the encoded parameters.
  setName(Name.c_str());
}

//===----------------------------------------------------------------------===//
// DWARF namespace (DW_TAG_namespace).
//===----------------------------------------------------------------------===//
void LVScopeNamespace::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind()) << " " << formattedName(getName()) << "\n";

  if (Full) {
    if (LVScope *Reference = getReference())
      Reference->printReference(OS, Full, const_cast<LVScopeNamespace *>(this));
  }
}

void LVScopeRoot::print(raw_ostream &OS, bool Full) const {
  OS << "\nLogical View:\n";
  LVScope::print(OS, Full);
}

void LVScopeRoot::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind()) << " " << formattedName(getName()) << "";
  if (options().getAttributeFormat())
    OS << " -> " << getFileFormatName();
  OS << "\n";
}

Error LVScopeRoot::doPrintMatches(bool Split, raw_ostream &OS,
                                  bool UseMatchedElements) const {
  // During a view output splitting, use the output stream created by the
  // split context, then switch to the reader output stream.
  static raw_ostream *StreamSplit = &OS;

  if (Scopes) {
    if (UseMatchedElements)
      options().resetPrintFormatting();
    print(OS);

    for (LVScope *Scope : *Scopes) {
      getReader().setCompileUnit(const_cast<LVScope *>(Scope));

      // If 'Split', we use the scope name (CU name) as the ouput file; the
      // delimiters in the pathname, must be replaced by a normal character.
      if (Split) {
        std::string ScopeName(Scope->getName());
        if (std::error_code EC =
                getReaderSplitContext().open(ScopeName, ".txt", OS))
          return createStringError(EC, "Unable to create split output file %s",
                                   ScopeName.c_str());
        StreamSplit = static_cast<raw_ostream *>(&getReaderSplitContext().os());
      }

      Scope->printMatchedElements(*StreamSplit, UseMatchedElements);

      // Done printing the compile unit. Restore the original output context.
      if (Split) {
        getReaderSplitContext().close();
        StreamSplit = &getReader().outputStream();
      }
    }
    if (UseMatchedElements)
      options().setPrintFormatting();
  }

  return Error::success();
}

//===----------------------------------------------------------------------===//
// DWARF template parameter pack (DW_TAG_GNU_template_parameter_pack).
//===----------------------------------------------------------------------===//
void LVScopeTemplatePack::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind()) << " " << formattedName(getName()) << "\n";
}
