//===-- LVReader.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the LVReader class.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"
#include "llvm/DebugInfo/LogicalView/Core/LVScope.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include <tuple>

using namespace llvm;
using namespace llvm::logicalview;

#define DEBUG_TYPE "Reader"

// Detect elements that are inserted more than once at different scopes,
// causing a crash on the reader destruction, as the element is already
// deleted from other scope. Helper for CodeView reader.
bool checkIntegrityScopesTree(LVScope *Root) {
  using LVDuplicateEntry = std::tuple<LVElement *, LVScope *, LVScope *>;
  using LVDuplicate = std::vector<LVDuplicateEntry>;
  LVDuplicate Duplicate;

  using LVIntegrity = std::map<LVElement *, LVScope *>;
  LVIntegrity Integrity;

  // Add the given element to the integrity map.
  auto AddElement = [&](LVElement *Element, LVScope *Scope) {
    LVIntegrity::iterator Iter = Integrity.find(Element);
    if (Iter == Integrity.end())
      Integrity.emplace(Element, Scope);
    else
      // We found a duplicate.
      Duplicate.emplace_back(Element, Scope, Iter->second);
  };

  // Recursively add all the elements in the scope.
  std::function<void(LVScope * Parent)> TraverseScope = [&](LVScope *Parent) {
    auto Traverse = [&](const auto *Set) {
      if (Set)
        for (const auto &Entry : *Set)
          AddElement(Entry, Parent);
    };
    if (const LVScopes *Scopes = Parent->getScopes()) {
      for (LVScope *Scope : *Scopes) {
        AddElement(Scope, Parent);
        TraverseScope(Scope);
      }
    }
    Traverse(Parent->getSymbols());
    Traverse(Parent->getTypes());
    Traverse(Parent->getLines());
  };

  // Start traversing the scopes root and print any duplicates.
  TraverseScope(Root);
  bool PassIntegrity = true;
  if (Duplicate.size()) {
    llvm::stable_sort(Duplicate, [](const auto &l, const auto &r) {
      return std::get<0>(l)->getID() < std::get<0>(r)->getID();
    });

    auto PrintIndex = [](unsigned Index) {
      if (Index)
        dbgs() << format("%8d: ", Index);
      else
        dbgs() << format("%8c: ", ' ');
    };
    auto PrintElement = [&](LVElement *Element, unsigned Index = 0) {
      PrintIndex(Index);
      std::string ElementName(Element->getName());
      dbgs() << format("%15s ID=0x%08x '%s'\n", Element->kind(),
                       Element->getID(), ElementName.c_str());
    };

    std::string RootName(Root->getName());
    dbgs() << formatv("{0}\n", fmt_repeat('=', 72));
    dbgs() << format("Root: '%s'\nDuplicated elements: %d\n", RootName.c_str(),
                     Duplicate.size());
    dbgs() << formatv("{0}\n", fmt_repeat('=', 72));

    unsigned Index = 0;
    for (const LVDuplicateEntry &Entry : Duplicate) {
      LVElement *Element;
      LVScope *First;
      LVScope *Second;
      std::tie(Element, First, Second) = Entry;
      dbgs() << formatv("\n{0}\n", fmt_repeat('-', 72));
      PrintElement(Element, ++Index);
      PrintElement(First);
      PrintElement(Second);
      dbgs() << formatv("{0}\n", fmt_repeat('-', 72));
    }
    PassIntegrity = false;
  }
  return PassIntegrity;
}

//===----------------------------------------------------------------------===//
// Class to represent a split context.
//===----------------------------------------------------------------------===//
Error LVSplitContext::createSplitFolder(StringRef Where) {
  // The 'location' will represent the root directory for the output created
  // by the context. It will contain the different CUs files, that will be
  // extracted from a single ELF.
  Location = std::string(Where);

  // Add a trailing slash, if there is none.
  size_t Pos = Location.find_last_of('/');
  if (Location.length() != Pos + 1)
    Location.append("/");

  // Make sure the new directory exists, creating it if necessary.
  if (std::error_code EC = llvm::sys::fs::create_directories(Location))
    return createStringError(EC, "Error: could not create directory %s",
                             Location.c_str());

  return Error::success();
}

std::error_code LVSplitContext::open(std::string ContextName,
                                     std::string Extension, raw_ostream &OS) {
  assert(OutputFile == nullptr && "OutputFile already set.");

  // Transforms '/', '\', '.', ':' into '_'.
  std::string Name(flattenedFilePath(ContextName));
  Name.append(Extension);
  // Add the split context location folder name.
  if (!Location.empty())
    Name.insert(0, Location);

  std::error_code EC;
  OutputFile = std::make_unique<ToolOutputFile>(Name, EC, sys::fs::OF_None);
  if (EC)
    return EC;

  // Don't remove output file.
  OutputFile->keep();
  return std::error_code();
}

LVReader *CurrentReader = nullptr;
LVReader &LVReader::getInstance() {
  if (CurrentReader)
    return *CurrentReader;
  outs() << "Invalid instance reader.\n";
  llvm_unreachable("Invalid instance reader.");
}
void LVReader::setInstance(LVReader *Reader) { CurrentReader = Reader; }

Error LVReader::createSplitFolder() {
  if (OutputSplit) {
    // If the '--output=split' was specified, but no '--split-folder'
    // option, use the input file as base for the split location.
    if (options().getOutputFolder().empty())
      options().setOutputFolder(getFilename().str() + "_cus");

    SmallString<128> SplitFolder;
    SplitFolder = options().getOutputFolder();
    sys::fs::make_absolute(SplitFolder);

    // Return error if unable to create a split context location.
    if (Error Err = SplitContext.createSplitFolder(SplitFolder))
      return Err;

    OS << "\nSplit View Location: '" << SplitContext.getLocation() << "'\n";
  }

  return Error::success();
}

// Get the filename for given object.
StringRef LVReader::getFilename(LVObject *Object, size_t Index) const {
  // TODO: The current CodeView Reader implementation does not have support
  // for multiple compile units. Until we have a proper offset calculation,
  // check only in the current compile unit.
  if (CompileUnits.size()) {
    // Get Compile Unit for the given object.
    LVCompileUnits::const_iterator Iter =
        std::prev(CompileUnits.lower_bound(Object->getOffset()));
    if (Iter != CompileUnits.end())
      return Iter->second->getFilename(Index);
  }

  return CompileUnit ? CompileUnit->getFilename(Index) : StringRef();
}

void LVReader::addSectionRange(LVSectionIndex SectionIndex, LVScope *Scope) {
  LVRange *ScopesWithRanges = getSectionRanges(SectionIndex);
  ScopesWithRanges->addEntry(Scope);
}

void LVReader::addSectionRange(LVSectionIndex SectionIndex, LVScope *Scope,
                               LVAddress LowerAddress, LVAddress UpperAddress) {
  LVRange *ScopesWithRanges = getSectionRanges(SectionIndex);
  ScopesWithRanges->addEntry(Scope, LowerAddress, UpperAddress);
}

LVRange *LVReader::getSectionRanges(LVSectionIndex SectionIndex) {
  // Check if we already have a mapping for this section index.
  LVSectionRanges::iterator IterSection = SectionRanges.find(SectionIndex);
  if (IterSection == SectionRanges.end())
    IterSection =
        SectionRanges.emplace(SectionIndex, std::make_unique<LVRange>()).first;
  LVRange *Range = IterSection->second.get();
  assert(Range && "Range is null.");
  return Range;
}

LVElement *LVReader::createElement(dwarf::Tag Tag) {
  CurrentScope = nullptr;
  CurrentSymbol = nullptr;
  CurrentType = nullptr;
  CurrentRanges.clear();

  LLVM_DEBUG(
      { dbgs() << "\n[createElement] " << dwarf::TagString(Tag) << "\n"; });

  if (!options().getPrintSymbols()) {
    switch (Tag) {
    // As the command line options did not specify a request to print
    // logical symbols (--print=symbols or --print=all or --print=elements),
    // skip its creation.
    case dwarf::DW_TAG_formal_parameter:
    case dwarf::DW_TAG_unspecified_parameters:
    case dwarf::DW_TAG_member:
    case dwarf::DW_TAG_variable:
    case dwarf::DW_TAG_inheritance:
    case dwarf::DW_TAG_constant:
    case dwarf::DW_TAG_call_site_parameter:
    case dwarf::DW_TAG_GNU_call_site_parameter:
      return nullptr;
    default:
      break;
    }
  }

  switch (Tag) {
  // Types.
  case dwarf::DW_TAG_base_type:
    CurrentType = createType();
    CurrentType->setIsBase();
    if (options().getAttributeBase())
      CurrentType->setIncludeInPrint();
    return CurrentType;
  case dwarf::DW_TAG_const_type:
    CurrentType = createType();
    CurrentType->setIsConst();
    CurrentType->setName("const");
    return CurrentType;
  case dwarf::DW_TAG_enumerator:
    CurrentType = createTypeEnumerator();
    return CurrentType;
  case dwarf::DW_TAG_imported_declaration:
    CurrentType = createTypeImport();
    CurrentType->setIsImportDeclaration();
    return CurrentType;
  case dwarf::DW_TAG_imported_module:
    CurrentType = createTypeImport();
    CurrentType->setIsImportModule();
    return CurrentType;
  case dwarf::DW_TAG_pointer_type:
    CurrentType = createType();
    CurrentType->setIsPointer();
    CurrentType->setName("*");
    return CurrentType;
  case dwarf::DW_TAG_ptr_to_member_type:
    CurrentType = createType();
    CurrentType->setIsPointerMember();
    CurrentType->setName("*");
    return CurrentType;
  case dwarf::DW_TAG_reference_type:
    CurrentType = createType();
    CurrentType->setIsReference();
    CurrentType->setName("&");
    return CurrentType;
  case dwarf::DW_TAG_restrict_type:
    CurrentType = createType();
    CurrentType->setIsRestrict();
    CurrentType->setName("restrict");
    return CurrentType;
  case dwarf::DW_TAG_rvalue_reference_type:
    CurrentType = createType();
    CurrentType->setIsRvalueReference();
    CurrentType->setName("&&");
    return CurrentType;
  case dwarf::DW_TAG_subrange_type:
    CurrentType = createTypeSubrange();
    return CurrentType;
  case dwarf::DW_TAG_template_value_parameter:
    CurrentType = createTypeParam();
    CurrentType->setIsTemplateValueParam();
    return CurrentType;
  case dwarf::DW_TAG_template_type_parameter:
    CurrentType = createTypeParam();
    CurrentType->setIsTemplateTypeParam();
    return CurrentType;
  case dwarf::DW_TAG_GNU_template_template_param:
    CurrentType = createTypeParam();
    CurrentType->setIsTemplateTemplateParam();
    return CurrentType;
  case dwarf::DW_TAG_typedef:
    CurrentType = createTypeDefinition();
    return CurrentType;
  case dwarf::DW_TAG_unspecified_type:
    CurrentType = createType();
    CurrentType->setIsUnspecified();
    return CurrentType;
  case dwarf::DW_TAG_volatile_type:
    CurrentType = createType();
    CurrentType->setIsVolatile();
    CurrentType->setName("volatile");
    return CurrentType;

  // Symbols.
  case dwarf::DW_TAG_formal_parameter:
    CurrentSymbol = createSymbol();
    CurrentSymbol->setIsParameter();
    return CurrentSymbol;
  case dwarf::DW_TAG_unspecified_parameters:
    CurrentSymbol = createSymbol();
    CurrentSymbol->setIsUnspecified();
    CurrentSymbol->setName("...");
    return CurrentSymbol;
  case dwarf::DW_TAG_member:
    CurrentSymbol = createSymbol();
    CurrentSymbol->setIsMember();
    return CurrentSymbol;
  case dwarf::DW_TAG_variable:
    CurrentSymbol = createSymbol();
    CurrentSymbol->setIsVariable();
    return CurrentSymbol;
  case dwarf::DW_TAG_inheritance:
    CurrentSymbol = createSymbol();
    CurrentSymbol->setIsInheritance();
    return CurrentSymbol;
  case dwarf::DW_TAG_call_site_parameter:
  case dwarf::DW_TAG_GNU_call_site_parameter:
    CurrentSymbol = createSymbol();
    CurrentSymbol->setIsCallSiteParameter();
    return CurrentSymbol;
  case dwarf::DW_TAG_constant:
    CurrentSymbol = createSymbol();
    CurrentSymbol->setIsConstant();
    return CurrentSymbol;

  // Scopes.
  case dwarf::DW_TAG_catch_block:
    CurrentScope = createScope();
    CurrentScope->setIsCatchBlock();
    return CurrentScope;
  case dwarf::DW_TAG_lexical_block:
    CurrentScope = createScope();
    CurrentScope->setIsLexicalBlock();
    return CurrentScope;
  case dwarf::DW_TAG_try_block:
    CurrentScope = createScope();
    CurrentScope->setIsTryBlock();
    return CurrentScope;
  case dwarf::DW_TAG_compile_unit:
  case dwarf::DW_TAG_skeleton_unit:
    CurrentScope = createScopeCompileUnit();
    CompileUnit = static_cast<LVScopeCompileUnit *>(CurrentScope);
    return CurrentScope;
  case dwarf::DW_TAG_inlined_subroutine:
    CurrentScope = createScopeFunctionInlined();
    return CurrentScope;
  case dwarf::DW_TAG_namespace:
    CurrentScope = createScopeNamespace();
    return CurrentScope;
  case dwarf::DW_TAG_template_alias:
    CurrentScope = createScopeAlias();
    return CurrentScope;
  case dwarf::DW_TAG_array_type:
    CurrentScope = createScopeArray();
    return CurrentScope;
  case dwarf::DW_TAG_call_site:
  case dwarf::DW_TAG_GNU_call_site:
    CurrentScope = createScopeFunction();
    CurrentScope->setIsCallSite();
    return CurrentScope;
  case dwarf::DW_TAG_entry_point:
    CurrentScope = createScopeFunction();
    CurrentScope->setIsEntryPoint();
    return CurrentScope;
  case dwarf::DW_TAG_subprogram:
    CurrentScope = createScopeFunction();
    CurrentScope->setIsSubprogram();
    return CurrentScope;
  case dwarf::DW_TAG_subroutine_type:
    CurrentScope = createScopeFunctionType();
    return CurrentScope;
  case dwarf::DW_TAG_label:
    CurrentScope = createScopeFunction();
    CurrentScope->setIsLabel();
    return CurrentScope;
  case dwarf::DW_TAG_class_type:
    CurrentScope = createScopeAggregate();
    CurrentScope->setIsClass();
    return CurrentScope;
  case dwarf::DW_TAG_structure_type:
    CurrentScope = createScopeAggregate();
    CurrentScope->setIsStructure();
    return CurrentScope;
  case dwarf::DW_TAG_union_type:
    CurrentScope = createScopeAggregate();
    CurrentScope->setIsUnion();
    return CurrentScope;
  case dwarf::DW_TAG_enumeration_type:
    CurrentScope = createScopeEnumeration();
    return CurrentScope;
  case dwarf::DW_TAG_GNU_formal_parameter_pack:
    CurrentScope = createScopeFormalPack();
    return CurrentScope;
  case dwarf::DW_TAG_GNU_template_parameter_pack:
    CurrentScope = createScopeTemplatePack();
    return CurrentScope;
  case dwarf::DW_TAG_module:
    CurrentScope = createScopeModule();
    return CurrentScope;
  default:
    // Collect TAGs not implemented.
    if (options().getInternalTag() && Tag)
      CompileUnit->addDebugTag(Tag, CurrentOffset);
    break;
  }

  LLVM_DEBUG({
    dbgs() << "DWARF Tag not implemented: " << dwarf::TagString(Tag) << "\n";
  });

  return nullptr;
}

// The Reader is the module that creates the logical view using the debug
// information contained in the binary file specified in the command line.
// This is the main entry point for the Reader and performs the following
// steps:
// - Process any patterns collected from the '--select' options.
// - For each compile unit in the debug information:
//   * Create the logical elements (scopes, symbols, types, lines).
//   * Collect debug ranges and debug locations.
//   * Move the collected logical lines to their associated scopes.
// - Once all the compile units have been processed, traverse the scopes
//   tree in order to:
//   * Calculate symbol coverage.
//   * Detect invalid ranges and locations.
//   * "resolve" the logical elements. During this pass, the names and
//      file information are updated, to reflect any dependency with other
//     logical elements.
Error LVReader::doLoad() {
  // Set current Reader instance.
  setInstance(this);

  // Before any scopes creation, process any pattern specified by the
  // --select and --select-offsets options.
  patterns().addGenericPatterns(options().Select.Generic);
  patterns().addOffsetPatterns(options().Select.Offsets);

  // Add any specific element printing requests based on the element kind.
  patterns().addRequest(options().Select.Elements);
  patterns().addRequest(options().Select.Lines);
  patterns().addRequest(options().Select.Scopes);
  patterns().addRequest(options().Select.Symbols);
  patterns().addRequest(options().Select.Types);

  // Once we have processed the requests for any particular kind of elements,
  // we need to update the report options, in order to have a default value.
  patterns().updateReportOptions();

  // Delegate the scope tree creation to the specific reader.
  if (Error Err = createScopes())
    return Err;

  if (options().getInternalIntegrity() && !checkIntegrityScopesTree(Root))
    return llvm::make_error<StringError>("Duplicated elements in Scopes Tree",
                                         inconvertibleErrorCode());

  // Calculate symbol coverage and detect invalid debug locations and ranges.
  Root->processRangeInformation();

  // As the elements can depend on elements from a different compile unit,
  // information such as name and file/line source information needs to be
  // updated.
  Root->resolveElements();

  sortScopes();
  return Error::success();
}

// Default handler for a generic reader.
Error LVReader::doPrint() {
  // Set current Reader instance.
  setInstance(this);

  // Check for any '--report' request.
  if (options().getReportExecute()) {
    // Requested details.
    if (options().getReportList())
      if (Error Err = printMatchedElements(/*UseMatchedElements=*/true))
        return Err;
    // Requested only children.
    if (options().getReportChildren() && !options().getReportParents())
      if (Error Err = printMatchedElements(/*UseMatchedElements=*/false))
        return Err;
    // Requested (parents) or (parents and children).
    if (options().getReportParents() || options().getReportView())
      if (Error Err = printScopes())
        return Err;

    return Error::success();
  }

  return printScopes();
}

Error LVReader::printScopes() {
  if (bool DoPrint =
          (options().getPrintExecute() || options().getComparePrint())) {
    if (Error Err = createSplitFolder())
      return Err;

    // Start printing from the root.
    bool DoMatch = options().getSelectGenericPattern() ||
                   options().getSelectGenericKind() ||
                   options().getSelectOffsetPattern();
    return Root->doPrint(OutputSplit, DoMatch, DoPrint, OS);
  }

  return Error::success();
}

Error LVReader::printMatchedElements(bool UseMatchedElements) {
  if (Error Err = createSplitFolder())
    return Err;

  return Root->doPrintMatches(OutputSplit, OS, UseMatchedElements);
}

void LVReader::print(raw_ostream &OS) const {
  OS << "LVReader\n";
  LLVM_DEBUG(dbgs() << "PrintReader\n");
}
