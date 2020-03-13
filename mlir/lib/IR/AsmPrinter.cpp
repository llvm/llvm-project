//===- AsmPrinter.cpp - MLIR Assembly Printer Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MLIR AsmPrinter class, which is used to implement
// the various print() methods on the core IR objects.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SaveAndRestore.h"
using namespace mlir;
using namespace mlir::detail;

void Identifier::print(raw_ostream &os) const { os << str(); }

void Identifier::dump() const { print(llvm::errs()); }

void OperationName::print(raw_ostream &os) const { os << getStringRef(); }

void OperationName::dump() const { print(llvm::errs()); }

DialectAsmPrinter::~DialectAsmPrinter() {}

OpAsmPrinter::~OpAsmPrinter() {}

//===--------------------------------------------------------------------===//
// Operation OpAsm interface.
//===--------------------------------------------------------------------===//

/// The OpAsmOpInterface, see OpAsmInterface.td for more details.
#include "mlir/IR/OpAsmInterface.cpp.inc"

//===----------------------------------------------------------------------===//
// OpPrintingFlags
//===----------------------------------------------------------------------===//

static llvm::cl::opt<int> printElementsAttrWithHexIfLarger(
    "mlir-print-elementsattrs-with-hex-if-larger",
    llvm::cl::desc(
        "Print DenseElementsAttrs with a hex string that have "
        "more elements than the given upper limit (use -1 to disable)"),
    llvm::cl::init(100));

static llvm::cl::opt<unsigned> elideElementsAttrIfLarger(
    "mlir-elide-elementsattrs-if-larger",
    llvm::cl::desc("Elide ElementsAttrs with \"...\" that have "
                   "more elements than the given upper limit"));

static llvm::cl::opt<bool>
    printDebugInfoOpt("mlir-print-debuginfo",
                      llvm::cl::desc("Print debug info in MLIR output"),
                      llvm::cl::init(false));

static llvm::cl::opt<bool> printPrettyDebugInfoOpt(
    "mlir-pretty-debuginfo",
    llvm::cl::desc("Print pretty debug info in MLIR output"),
    llvm::cl::init(false));

// Use the generic op output form in the operation printer even if the custom
// form is defined.
static llvm::cl::opt<bool>
    printGenericOpFormOpt("mlir-print-op-generic",
                          llvm::cl::desc("Print the generic op form"),
                          llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<bool> printLocalScopeOpt(
    "mlir-print-local-scope",
    llvm::cl::desc("Print assuming in local scope by default"),
    llvm::cl::init(false), llvm::cl::Hidden);

/// Initialize the printing flags with default supplied by the cl::opts above.
OpPrintingFlags::OpPrintingFlags()
    : elementsAttrElementLimit(
          elideElementsAttrIfLarger.getNumOccurrences()
              ? Optional<int64_t>(elideElementsAttrIfLarger)
              : Optional<int64_t>()),
      printDebugInfoFlag(printDebugInfoOpt),
      printDebugInfoPrettyFormFlag(printPrettyDebugInfoOpt),
      printGenericOpFormFlag(printGenericOpFormOpt),
      printLocalScope(printLocalScopeOpt) {}

/// Enable the elision of large elements attributes, by printing a '...'
/// instead of the element data, when the number of elements is greater than
/// `largeElementLimit`. Note: The IR generated with this option is not
/// parsable.
OpPrintingFlags &
OpPrintingFlags::elideLargeElementsAttrs(int64_t largeElementLimit) {
  elementsAttrElementLimit = largeElementLimit;
  return *this;
}

/// Enable printing of debug information. If 'prettyForm' is set to true,
/// debug information is printed in a more readable 'pretty' form.
OpPrintingFlags &OpPrintingFlags::enableDebugInfo(bool prettyForm) {
  printDebugInfoFlag = true;
  printDebugInfoPrettyFormFlag = prettyForm;
  return *this;
}

/// Always print operations in the generic form.
OpPrintingFlags &OpPrintingFlags::printGenericOpForm() {
  printGenericOpFormFlag = true;
  return *this;
}

/// Use local scope when printing the operation. This allows for using the
/// printer in a more localized and thread-safe setting, but may not necessarily
/// be identical of what the IR will look like when dumping the full module.
OpPrintingFlags &OpPrintingFlags::useLocalScope() {
  printLocalScope = true;
  return *this;
}

/// Return if the given ElementsAttr should be elided.
bool OpPrintingFlags::shouldElideElementsAttr(ElementsAttr attr) const {
  return elementsAttrElementLimit.hasValue() &&
         *elementsAttrElementLimit < int64_t(attr.getNumElements());
}

/// Return if debug information should be printed.
bool OpPrintingFlags::shouldPrintDebugInfo() const {
  return printDebugInfoFlag;
}

/// Return if debug information should be printed in the pretty form.
bool OpPrintingFlags::shouldPrintDebugInfoPrettyForm() const {
  return printDebugInfoPrettyFormFlag;
}

/// Return if operations should be printed in the generic form.
bool OpPrintingFlags::shouldPrintGenericOpForm() const {
  return printGenericOpFormFlag;
}

/// Return if the printer should use local scope when dumping the IR.
bool OpPrintingFlags::shouldUseLocalScope() const { return printLocalScope; }

//===----------------------------------------------------------------------===//
// NewLineCounter
//===----------------------------------------------------------------------===//

namespace {
/// This class is a simple formatter that emits a new line when inputted into a
/// stream, that enables counting the number of newlines emitted. This class
/// should be used whenever emitting newlines in the printer.
struct NewLineCounter {
  unsigned curLine = 1;
};
} // end anonymous namespace

static raw_ostream &operator<<(raw_ostream &os, NewLineCounter &newLine) {
  ++newLine.curLine;
  return os << '\n';
}

//===----------------------------------------------------------------------===//
// AliasState
//===----------------------------------------------------------------------===//

namespace {
/// This class manages the state for type and attribute aliases.
class AliasState {
public:
  // Initialize the internal aliases.
  void
  initialize(Operation *op,
             DialectInterfaceCollection<OpAsmDialectInterface> &interfaces);

  /// Return a name used for an attribute alias, or empty if there is no alias.
  Twine getAttributeAlias(Attribute attr) const;

  /// Print all of the referenced attribute aliases.
  void printAttributeAliases(raw_ostream &os, NewLineCounter &newLine) const;

  /// Return a string to use as an alias for the given type, or empty if there
  /// is no alias recorded.
  StringRef getTypeAlias(Type ty) const;

  /// Print all of the referenced type aliases.
  void printTypeAliases(raw_ostream &os, NewLineCounter &newLine) const;

private:
  /// A special index constant used for non-kind attribute aliases.
  enum { NonAttrKindAlias = -1 };

  /// Record a reference to the given attribute.
  void recordAttributeReference(Attribute attr);

  /// Record a reference to the given type.
  void recordTypeReference(Type ty);

  // Visit functions.
  void visitOperation(Operation *op);
  void visitType(Type type);
  void visitAttribute(Attribute attr);

  /// Set of attributes known to be used within the module.
  llvm::SetVector<Attribute> usedAttributes;

  /// Mapping between attribute and a pair comprised of a base alias name and a
  /// count suffix. If the suffix is set to -1, it is not displayed.
  llvm::MapVector<Attribute, std::pair<StringRef, int>> attrToAlias;

  /// Mapping between attribute kind and a pair comprised of a base alias name
  /// and a unique list of attributes belonging to this kind sorted by location
  /// seen in the module.
  llvm::MapVector<unsigned, std::pair<StringRef, std::vector<Attribute>>>
      attrKindToAlias;

  /// Set of types known to be used within the module.
  llvm::SetVector<Type> usedTypes;

  /// A mapping between a type and a given alias.
  DenseMap<Type, StringRef> typeToAlias;
};
} // end anonymous namespace

// Utility to generate a function to register a symbol alias.
static bool canRegisterAlias(StringRef name, llvm::StringSet<> &usedAliases) {
  assert(!name.empty() && "expected alias name to be non-empty");
  // TODO(riverriddle) Assert that the provided alias name can be lexed as
  // an identifier.

  // Check that the alias doesn't contain a '.' character and the name is not
  // already in use.
  return !name.contains('.') && usedAliases.insert(name).second;
}

void AliasState::initialize(
    Operation *op,
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces) {
  // Track the identifiers in use for each symbol so that the same identifier
  // isn't used twice.
  llvm::StringSet<> usedAliases;

  // Collect the set of aliases from each dialect.
  SmallVector<std::pair<unsigned, StringRef>, 8> attributeKindAliases;
  SmallVector<std::pair<Attribute, StringRef>, 8> attributeAliases;
  SmallVector<std::pair<Type, StringRef>, 16> typeAliases;

  // AffineMap/Integer set have specific kind aliases.
  attributeKindAliases.emplace_back(StandardAttributes::AffineMap, "map");
  attributeKindAliases.emplace_back(StandardAttributes::IntegerSet, "set");

  for (auto &interface : interfaces) {
    interface.getAttributeKindAliases(attributeKindAliases);
    interface.getAttributeAliases(attributeAliases);
    interface.getTypeAliases(typeAliases);
  }

  // Setup the attribute kind aliases.
  StringRef alias;
  unsigned attrKind;
  for (auto &attrAliasPair : attributeKindAliases) {
    std::tie(attrKind, alias) = attrAliasPair;
    assert(!alias.empty() && "expected non-empty alias string");
    if (!usedAliases.count(alias) && !alias.contains('.'))
      attrKindToAlias.insert({attrKind, {alias, {}}});
  }

  // Clear the set of used identifiers so that the attribute kind aliases are
  // just a prefix and not the full alias, i.e. there may be some overlap.
  usedAliases.clear();

  // Register the attribute aliases.
  // Create a regex for the attribute kind alias names, these have a prefix with
  // a counter appended to the end. We prevent normal aliases from having these
  // names to avoid collisions.
  llvm::Regex reservedAttrNames("[0-9]+$");

  // Attribute value aliases.
  Attribute attr;
  for (auto &attrAliasPair : attributeAliases) {
    std::tie(attr, alias) = attrAliasPair;
    if (!reservedAttrNames.match(alias) && canRegisterAlias(alias, usedAliases))
      attrToAlias.insert({attr, {alias, NonAttrKindAlias}});
  }

  // Clear the set of used identifiers as types can have the same identifiers as
  // affine structures.
  usedAliases.clear();

  // Type aliases.
  for (auto &typeAliasPair : typeAliases)
    if (canRegisterAlias(typeAliasPair.second, usedAliases))
      typeToAlias.insert(typeAliasPair);

  // Traverse the given IR to generate the set of used attributes/types.
  op->walk([&](Operation *op) { visitOperation(op); });
}

/// Return a name used for an attribute alias, or empty if there is no alias.
Twine AliasState::getAttributeAlias(Attribute attr) const {
  auto alias = attrToAlias.find(attr);
  if (alias == attrToAlias.end())
    return Twine();

  // Return the alias for this attribute, along with the index if this was
  // generated by a kind alias.
  int kindIndex = alias->second.second;
  return alias->second.first +
         (kindIndex == NonAttrKindAlias ? Twine() : Twine(kindIndex));
}

/// Print all of the referenced attribute aliases.
void AliasState::printAttributeAliases(raw_ostream &os,
                                       NewLineCounter &newLine) const {
  auto printAlias = [&](StringRef alias, Attribute attr, int index) {
    os << '#' << alias;
    if (index != NonAttrKindAlias)
      os << index;
    os << " = " << attr << newLine;
  };

  // Print all of the attribute kind aliases.
  for (auto &kindAlias : attrKindToAlias) {
    auto &aliasAttrsPair = kindAlias.second;
    for (unsigned i = 0, e = aliasAttrsPair.second.size(); i != e; ++i)
      printAlias(aliasAttrsPair.first, aliasAttrsPair.second[i], i);
    os << newLine;
  }

  // In a second pass print all of the remaining attribute aliases that aren't
  // kind aliases.
  for (Attribute attr : usedAttributes) {
    auto alias = attrToAlias.find(attr);
    if (alias != attrToAlias.end() && alias->second.second == NonAttrKindAlias)
      printAlias(alias->second.first, attr, alias->second.second);
  }
}

/// Return a string to use as an alias for the given type, or empty if there
/// is no alias recorded.
StringRef AliasState::getTypeAlias(Type ty) const {
  return typeToAlias.lookup(ty);
}

/// Print all of the referenced type aliases.
void AliasState::printTypeAliases(raw_ostream &os,
                                  NewLineCounter &newLine) const {
  for (Type type : usedTypes) {
    auto alias = typeToAlias.find(type);
    if (alias != typeToAlias.end())
      os << '!' << alias->second << " = type " << type << newLine;
  }
}

/// Record a reference to the given attribute.
void AliasState::recordAttributeReference(Attribute attr) {
  // Don't recheck attributes that have already been seen or those that
  // already have an alias.
  if (!usedAttributes.insert(attr) || attrToAlias.count(attr))
    return;

  // If this attribute kind has an alias, then record one for this attribute.
  auto alias = attrKindToAlias.find(static_cast<unsigned>(attr.getKind()));
  if (alias == attrKindToAlias.end())
    return;
  std::pair<StringRef, int> attrAlias(alias->second.first,
                                      alias->second.second.size());
  attrToAlias.insert({attr, attrAlias});
  alias->second.second.push_back(attr);
}

/// Record a reference to the given type.
void AliasState::recordTypeReference(Type ty) { usedTypes.insert(ty); }

// TODO Support visiting other types/operations when implemented.
void AliasState::visitType(Type type) {
  recordTypeReference(type);

  if (auto funcType = type.dyn_cast<FunctionType>()) {
    // Visit input and result types for functions.
    for (auto input : funcType.getInputs())
      visitType(input);
    for (auto result : funcType.getResults())
      visitType(result);
  } else if (auto shapedType = type.dyn_cast<ShapedType>()) {
    visitType(shapedType.getElementType());

    // Visit affine maps in memref type.
    if (auto memref = type.dyn_cast<MemRefType>())
      for (auto map : memref.getAffineMaps())
        recordAttributeReference(AffineMapAttr::get(map));
  }
}

void AliasState::visitAttribute(Attribute attr) {
  recordAttributeReference(attr);

  if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
    for (auto elt : arrayAttr.getValue())
      visitAttribute(elt);
  } else if (auto typeAttr = attr.dyn_cast<TypeAttr>()) {
    visitType(typeAttr.getValue());
  }
}

void AliasState::visitOperation(Operation *op) {
  // Visit all the types used in the operation.
  for (auto type : op->getOperandTypes())
    visitType(type);
  for (auto type : op->getResultTypes())
    visitType(type);
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto arg : block.getArguments())
        visitType(arg.getType());

  // Visit each of the attributes.
  for (auto elt : op->getAttrs())
    visitAttribute(elt.second);
}

//===----------------------------------------------------------------------===//
// SSANameState
//===----------------------------------------------------------------------===//

namespace {
/// This class manages the state of SSA value names.
class SSANameState {
public:
  /// A sentinel value used for values with names set.
  enum : unsigned { NameSentinel = ~0U };

  SSANameState(Operation *op,
               DialectInterfaceCollection<OpAsmDialectInterface> &interfaces);

  /// Print the SSA identifier for the given value to 'stream'. If
  /// 'printResultNo' is true, it also presents the result number ('#' number)
  /// of this value.
  void printValueID(Value value, bool printResultNo, raw_ostream &stream) const;

  /// Return the result indices for each of the result groups registered by this
  /// operation, or empty if none exist.
  ArrayRef<int> getOpResultGroups(Operation *op);

  /// Get the ID for the given block.
  unsigned getBlockID(Block *block);

  /// Renumber the arguments for the specified region to the same names as the
  /// SSA values in namesToUse. See OperationPrinter::shadowRegionArgs for
  /// details.
  void shadowRegionArgs(Region &region, ValueRange namesToUse);

private:
  /// Number the SSA values within the given IR unit.
  void numberValuesInRegion(
      Region &region,
      DialectInterfaceCollection<OpAsmDialectInterface> &interfaces);
  void numberValuesInBlock(
      Block &block,
      DialectInterfaceCollection<OpAsmDialectInterface> &interfaces);
  void numberValuesInOp(
      Operation &op,
      DialectInterfaceCollection<OpAsmDialectInterface> &interfaces);

  /// Given a result of an operation 'result', find the result group head
  /// 'lookupValue' and the result of 'result' within that group in
  /// 'lookupResultNo'. 'lookupResultNo' is only filled in if the result group
  /// has more than 1 result.
  void getResultIDAndNumber(OpResult result, Value &lookupValue,
                            Optional<int> &lookupResultNo) const;

  /// Set a special value name for the given value.
  void setValueName(Value value, StringRef name);

  /// Uniques the given value name within the printer. If the given name
  /// conflicts, it is automatically renamed.
  StringRef uniqueValueName(StringRef name);

  /// This is the value ID for each SSA value. If this returns NameSentinel,
  /// then the valueID has an entry in valueNames.
  DenseMap<Value, unsigned> valueIDs;
  DenseMap<Value, StringRef> valueNames;

  /// This is a map of operations that contain multiple named result groups,
  /// i.e. there may be multiple names for the results of the operation. The
  /// value of this map are the result numbers that start a result group.
  DenseMap<Operation *, SmallVector<int, 1>> opResultGroups;

  /// This is the block ID for each block in the current.
  DenseMap<Block *, unsigned> blockIDs;

  /// This keeps track of all of the non-numeric names that are in flight,
  /// allowing us to check for duplicates.
  /// Note: the value of the map is unused.
  llvm::ScopedHashTable<StringRef, char> usedNames;
  llvm::BumpPtrAllocator usedNameAllocator;

  /// This is the next value ID to assign in numbering.
  unsigned nextValueID = 0;
  /// This is the next ID to assign to a region entry block argument.
  unsigned nextArgumentID = 0;
  /// This is the next ID to assign when a name conflict is detected.
  unsigned nextConflictID = 0;
};
} // end anonymous namespace

SSANameState::SSANameState(
    Operation *op,
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces) {
  llvm::ScopedHashTable<StringRef, char>::ScopeTy usedNamesScope(usedNames);
  numberValuesInOp(*op, interfaces);

  for (auto &region : op->getRegions())
    numberValuesInRegion(region, interfaces);
}

void SSANameState::printValueID(Value value, bool printResultNo,
                                raw_ostream &stream) const {
  if (!value) {
    stream << "<<NULL>>";
    return;
  }

  Optional<int> resultNo;
  auto lookupValue = value;

  // If this is an operation result, collect the head lookup value of the result
  // group and the result number of 'result' within that group.
  if (OpResult result = value.dyn_cast<OpResult>())
    getResultIDAndNumber(result, lookupValue, resultNo);

  auto it = valueIDs.find(lookupValue);
  if (it == valueIDs.end()) {
    stream << "<<UNKNOWN SSA VALUE>>";
    return;
  }

  stream << '%';
  if (it->second != NameSentinel) {
    stream << it->second;
  } else {
    auto nameIt = valueNames.find(lookupValue);
    assert(nameIt != valueNames.end() && "Didn't have a name entry?");
    stream << nameIt->second;
  }

  if (resultNo.hasValue() && printResultNo)
    stream << '#' << resultNo;
}

ArrayRef<int> SSANameState::getOpResultGroups(Operation *op) {
  auto it = opResultGroups.find(op);
  return it == opResultGroups.end() ? ArrayRef<int>() : it->second;
}

unsigned SSANameState::getBlockID(Block *block) {
  auto it = blockIDs.find(block);
  return it != blockIDs.end() ? it->second : NameSentinel;
}

void SSANameState::shadowRegionArgs(Region &region, ValueRange namesToUse) {
  assert(!region.empty() && "cannot shadow arguments of an empty region");
  assert(region.front().getNumArguments() == namesToUse.size() &&
         "incorrect number of names passed in");
  assert(region.getParentOp()->isKnownIsolatedFromAbove() &&
         "only KnownIsolatedFromAbove ops can shadow names");

  SmallVector<char, 16> nameStr;
  for (unsigned i = 0, e = namesToUse.size(); i != e; ++i) {
    auto nameToUse = namesToUse[i];
    if (nameToUse == nullptr)
      continue;
    auto nameToReplace = region.front().getArgument(i);

    nameStr.clear();
    llvm::raw_svector_ostream nameStream(nameStr);
    printValueID(nameToUse, /*printResultNo=*/true, nameStream);

    // Entry block arguments should already have a pretty "arg" name.
    assert(valueIDs[nameToReplace] == NameSentinel);

    // Use the name without the leading %.
    auto name = StringRef(nameStream.str()).drop_front();

    // Overwrite the name.
    valueNames[nameToReplace] = name.copy(usedNameAllocator);
  }
}

void SSANameState::numberValuesInRegion(
    Region &region,
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces) {
  // Save the current value ids to allow for numbering values in sibling regions
  // the same.
  llvm::SaveAndRestore<unsigned> valueIDSaver(nextValueID);
  llvm::SaveAndRestore<unsigned> argumentIDSaver(nextArgumentID);
  llvm::SaveAndRestore<unsigned> conflictIDSaver(nextConflictID);

  // Push a new used names scope.
  llvm::ScopedHashTable<StringRef, char>::ScopeTy usedNamesScope(usedNames);

  // Number the values within this region in a breadth-first order.
  unsigned nextBlockID = 0;
  for (auto &block : region) {
    // Each block gets a unique ID, and all of the operations within it get
    // numbered as well.
    blockIDs[&block] = nextBlockID++;
    numberValuesInBlock(block, interfaces);
  }

  // After that we traverse the nested regions.
  // TODO: Rework this loop to not use recursion.
  for (auto &block : region) {
    for (auto &op : block)
      for (auto &nestedRegion : op.getRegions())
        numberValuesInRegion(nestedRegion, interfaces);
  }
}

void SSANameState::numberValuesInBlock(
    Block &block,
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces) {
  auto setArgNameFn = [&](Value arg, StringRef name) {
    assert(!valueIDs.count(arg) && "arg numbered multiple times");
    assert(arg.cast<BlockArgument>().getOwner() == &block &&
           "arg not defined in 'block'");
    setValueName(arg, name);
  };

  bool isEntryBlock = block.isEntryBlock();
  if (isEntryBlock) {
    if (auto *op = block.getParentOp()) {
      if (auto asmInterface = interfaces.getInterfaceFor(op->getDialect()))
        asmInterface->getAsmBlockArgumentNames(&block, setArgNameFn);
    }
  }

  // Number the block arguments. We give entry block arguments a special name
  // 'arg'.
  SmallString<32> specialNameBuffer(isEntryBlock ? "arg" : "");
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  for (auto arg : block.getArguments()) {
    if (valueIDs.count(arg))
      continue;
    if (isEntryBlock) {
      specialNameBuffer.resize(strlen("arg"));
      specialName << nextArgumentID++;
    }
    setValueName(arg, specialName.str());
  }

  // Number the operations in this block.
  for (auto &op : block)
    numberValuesInOp(op, interfaces);
}

void SSANameState::numberValuesInOp(
    Operation &op,
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces) {
  unsigned numResults = op.getNumResults();
  if (numResults == 0)
    return;
  Value resultBegin = op.getResult(0);

  // Function used to set the special result names for the operation.
  SmallVector<int, 2> resultGroups(/*Size=*/1, /*Value=*/0);
  auto setResultNameFn = [&](Value result, StringRef name) {
    assert(!valueIDs.count(result) && "result numbered multiple times");
    assert(result.getDefiningOp() == &op && "result not defined by 'op'");
    setValueName(result, name);

    // Record the result number for groups not anchored at 0.
    if (int resultNo = result.cast<OpResult>().getResultNumber())
      resultGroups.push_back(resultNo);
  };
  if (OpAsmOpInterface asmInterface = dyn_cast<OpAsmOpInterface>(&op))
    asmInterface.getAsmResultNames(setResultNameFn);
  else if (auto *asmInterface = interfaces.getInterfaceFor(op.getDialect()))
    asmInterface->getAsmResultNames(&op, setResultNameFn);

  // If the first result wasn't numbered, give it a default number.
  if (valueIDs.try_emplace(resultBegin, nextValueID).second)
    ++nextValueID;

  // If this operation has multiple result groups, mark it.
  if (resultGroups.size() != 1) {
    llvm::array_pod_sort(resultGroups.begin(), resultGroups.end());
    opResultGroups.try_emplace(&op, std::move(resultGroups));
  }
}

void SSANameState::getResultIDAndNumber(OpResult result, Value &lookupValue,
                                        Optional<int> &lookupResultNo) const {
  Operation *owner = result.getOwner();
  if (owner->getNumResults() == 1)
    return;
  int resultNo = result.getResultNumber();

  // If this operation has multiple result groups, we will need to find the
  // one corresponding to this result.
  auto resultGroupIt = opResultGroups.find(owner);
  if (resultGroupIt == opResultGroups.end()) {
    // If not, just use the first result.
    lookupResultNo = resultNo;
    lookupValue = owner->getResult(0);
    return;
  }

  // Find the correct index using a binary search, as the groups are ordered.
  ArrayRef<int> resultGroups = resultGroupIt->second;
  auto it = llvm::upper_bound(resultGroups, resultNo);
  int groupResultNo = 0, groupSize = 0;

  // If there are no smaller elements, the last result group is the lookup.
  if (it == resultGroups.end()) {
    groupResultNo = resultGroups.back();
    groupSize = static_cast<int>(owner->getNumResults()) - resultGroups.back();
  } else {
    // Otherwise, the previous element is the lookup.
    groupResultNo = *std::prev(it);
    groupSize = *it - groupResultNo;
  }

  // We only record the result number for a group of size greater than 1.
  if (groupSize != 1)
    lookupResultNo = resultNo - groupResultNo;
  lookupValue = owner->getResult(groupResultNo);
}

void SSANameState::setValueName(Value value, StringRef name) {
  // If the name is empty, the value uses the default numbering.
  if (name.empty()) {
    valueIDs[value] = nextValueID++;
    return;
  }

  valueIDs[value] = NameSentinel;
  valueNames[value] = uniqueValueName(name);
}

/// Returns true if 'c' is an allowable punctuation character: [$._-]
/// Returns false otherwise.
static bool isPunct(char c) {
  return c == '$' || c == '.' || c == '_' || c == '-';
}
  
StringRef SSANameState::uniqueValueName(StringRef name) {
  assert(!name.empty() && "Shouldn't have an empty name here");
  
  // Check to see if this name is valid.  If it starts with a digit, then it
  // could conflict with the autogenerated numeric ID's (we unique them in a
  // different map), so add an underscore prefix to avoid problems.
  if (isdigit(name[0])) {
    SmallString<16> tmpName("_");
    tmpName += name;
    return uniqueValueName(tmpName);
  }
  
  // Check to see if the name consists of all-valid identifiers.  If not, we
  // need to escape them.
  for (char ch : name) {
    if (isalpha(ch) || isPunct(ch) || isdigit(ch))
      continue;
    
    SmallString<16> tmpName;
    for (char ch : name) {
      if (isalpha(ch) || isPunct(ch) || isdigit(ch))
        tmpName += ch;
      else if (ch == ' ')
        tmpName += '_';
      else {
        tmpName += llvm::utohexstr((unsigned char)ch);
      }
    }
    return uniqueValueName(tmpName);
  }
  
  // Check to see if this name is already unique.
  if (!usedNames.count(name)) {
    name = name.copy(usedNameAllocator);
  } else {
    // Otherwise, we had a conflict - probe until we find a unique name. This
    // is guaranteed to terminate (and usually in a single iteration) because it
    // generates new names by incrementing nextConflictID.
    SmallString<64> probeName(name);
    probeName.push_back('_');
    while (true) {
      probeName.resize(name.size() + 1);
      probeName += llvm::utostr(nextConflictID++);
      if (!usedNames.count(probeName)) {
        name = StringRef(probeName).copy(usedNameAllocator);
        break;
      }
    }
  }

  usedNames.insert(name, char());
  return name;
}

//===----------------------------------------------------------------------===//
// AsmState
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
class AsmStateImpl {
public:
  explicit AsmStateImpl(Operation *op, AsmState::LocationMap *locationMap)
      : interfaces(op->getContext()), nameState(op, interfaces),
        locationMap(locationMap) {}

  /// Initialize the alias state to enable the printing of aliases.
  void initializeAliases(Operation *op) {
    aliasState.initialize(op, interfaces);
  }

  /// Get an instance of the OpAsmDialectInterface for the given dialect, or
  /// null if one wasn't registered.
  const OpAsmDialectInterface *getOpAsmInterface(Dialect *dialect) {
    return interfaces.getInterfaceFor(dialect);
  }

  /// Get the state used for aliases.
  AliasState &getAliasState() { return aliasState; }

  /// Get the state used for SSA names.
  SSANameState &getSSANameState() { return nameState; }

  /// Register the location, line and column, within the buffer that the given
  /// operation was printed at.
  void registerOperationLocation(Operation *op, unsigned line, unsigned col) {
    if (locationMap)
      (*locationMap)[op] = std::make_pair(line, col);
  }

private:
  /// Collection of OpAsm interfaces implemented in the context.
  DialectInterfaceCollection<OpAsmDialectInterface> interfaces;

  /// The state used for attribute and type aliases.
  AliasState aliasState;

  /// The state used for SSA value names.
  SSANameState nameState;

  /// An optional location map to be populated.
  AsmState::LocationMap *locationMap;
};
} // end namespace detail
} // end namespace mlir

AsmState::AsmState(Operation *op, LocationMap *locationMap)
    : impl(std::make_unique<AsmStateImpl>(op, locationMap)) {}
AsmState::~AsmState() {}

//===----------------------------------------------------------------------===//
// ModulePrinter
//===----------------------------------------------------------------------===//

namespace {
class ModulePrinter {
public:
  ModulePrinter(raw_ostream &os, OpPrintingFlags flags = llvm::None,
                AsmStateImpl *state = nullptr)
      : os(os), printerFlags(flags), state(state) {}
  explicit ModulePrinter(ModulePrinter &printer)
      : os(printer.os), printerFlags(printer.printerFlags),
        state(printer.state) {}

  /// Returns the output stream of the printer.
  raw_ostream &getStream() { return os; }

  template <typename Container, typename UnaryFunctor>
  inline void interleaveComma(const Container &c, UnaryFunctor each_fn) const {
    mlir::interleaveComma(c, os, each_fn);
  }

  /// This enum descripes the different kinds of elision for the type of an
  /// attribute when printing it.
  enum class AttrTypeElision {
    /// The type must not be elided,
    Never,
    /// The type may be elided when it matches the default used in the parser
    /// (for example i64 is the default for integer attributes).
    May,
    /// The type must be elided.
    Must
  };

  /// Print the given attribute.
  void printAttribute(Attribute attr,
                      AttrTypeElision typeElision = AttrTypeElision::Never);

  void printType(Type type);
  void printLocation(LocationAttr loc);

  void printAffineMap(AffineMap map);
  void
  printAffineExpr(AffineExpr expr,
                  function_ref<void(unsigned, bool)> printValueName = nullptr);
  void printAffineConstraint(AffineExpr expr, bool isEq);
  void printIntegerSet(IntegerSet set);

protected:
  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<StringRef> elidedAttrs = {},
                             bool withKeyword = false);
  void printNamedAttribute(NamedAttribute attr);
  void printTrailingLocation(Location loc);
  void printLocationInternal(LocationAttr loc, bool pretty = false);

  /// Print a dense elements attribute. If 'allowHex' is true, a hex string is
  /// used instead of individual elements when the elements attr is large.
  void printDenseElementsAttr(DenseElementsAttr attr, bool allowHex);

  void printDialectAttribute(Attribute attr);
  void printDialectType(Type type);

  /// This enum is used to represent the binding strength of the enclosing
  /// context that an AffineExprStorage is being printed in, so we can
  /// intelligently produce parens.
  enum class BindingStrength {
    Weak,   // + and -
    Strong, // All other binary operators.
  };
  void printAffineExprInternal(
      AffineExpr expr, BindingStrength enclosingTightness,
      function_ref<void(unsigned, bool)> printValueName = nullptr);

  /// The output stream for the printer.
  raw_ostream &os;

  /// A set of flags to control the printer's behavior.
  OpPrintingFlags printerFlags;

  /// An optional printer state for the module.
  AsmStateImpl *state;

  /// A tracker for the number of new lines emitted during printing.
  NewLineCounter newLine;
};
} // end anonymous namespace

void ModulePrinter::printTrailingLocation(Location loc) {
  // Check to see if we are printing debug information.
  if (!printerFlags.shouldPrintDebugInfo())
    return;

  os << " ";
  printLocation(loc);
}

void ModulePrinter::printLocationInternal(LocationAttr loc, bool pretty) {
  switch (loc.getKind()) {
  case StandardAttributes::OpaqueLocation:
    printLocationInternal(loc.cast<OpaqueLoc>().getFallbackLocation(), pretty);
    break;
  case StandardAttributes::UnknownLocation:
    if (pretty)
      os << "[unknown]";
    else
      os << "unknown";
    break;
  case StandardAttributes::FileLineColLocation: {
    auto fileLoc = loc.cast<FileLineColLoc>();
    auto mayQuote = pretty ? "" : "\"";
    os << mayQuote << fileLoc.getFilename() << mayQuote << ':'
       << fileLoc.getLine() << ':' << fileLoc.getColumn();
    break;
  }
  case StandardAttributes::NameLocation: {
    auto nameLoc = loc.cast<NameLoc>();
    os << '\"' << nameLoc.getName() << '\"';

    // Print the child if it isn't unknown.
    auto childLoc = nameLoc.getChildLoc();
    if (!childLoc.isa<UnknownLoc>()) {
      os << '(';
      printLocationInternal(childLoc, pretty);
      os << ')';
    }
    break;
  }
  case StandardAttributes::CallSiteLocation: {
    auto callLocation = loc.cast<CallSiteLoc>();
    auto caller = callLocation.getCaller();
    auto callee = callLocation.getCallee();
    if (!pretty)
      os << "callsite(";
    printLocationInternal(callee, pretty);
    if (pretty) {
      if (callee.isa<NameLoc>()) {
        if (caller.isa<FileLineColLoc>()) {
          os << " at ";
        } else {
          os << newLine << " at ";
        }
      } else {
        os << newLine << " at ";
      }
    } else {
      os << " at ";
    }
    printLocationInternal(caller, pretty);
    if (!pretty)
      os << ")";
    break;
  }
  case StandardAttributes::FusedLocation: {
    auto fusedLoc = loc.cast<FusedLoc>();
    if (!pretty)
      os << "fused";
    if (auto metadata = fusedLoc.getMetadata())
      os << '<' << metadata << '>';
    os << '[';
    interleave(
        fusedLoc.getLocations(),
        [&](Location loc) { printLocationInternal(loc, pretty); },
        [&]() { os << ", "; });
    os << ']';
    break;
  }
  }
}

/// Print a floating point value in a way that the parser will be able to
/// round-trip losslessly.
static void printFloatValue(const APFloat &apValue, raw_ostream &os) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  bool isInf = apValue.isInfinity();
  bool isNaN = apValue.isNaN();
  if (!isInf && !isNaN) {
    SmallString<128> strValue;
    apValue.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                     /*TruncateZero=*/false);

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
    // that the string matches the "[-+]?[0-9]" regex.
    assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
            ((strValue[0] == '-' || strValue[0] == '+') &&
             (strValue[1] >= '0' && strValue[1] <= '9'))) &&
           "[-+]?[0-9] regex does not match!");

    // Parse back the stringized version and check that the value is equal
    // (i.e., there is no precision loss).
    if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
      os << strValue;
      return;
    }

    // If it is not, use the default format of APFloat instead of the
    // exponential notation.
    strValue.clear();
    apValue.toString(strValue);

    // Make sure that we can parse the default form as a float.
    if (StringRef(strValue).contains('.')) {
      os << strValue;
      return;
    }
  }

  // Print special values in hexadecimal format. The sign bit should be included
  // in the literal.
  SmallVector<char, 16> str;
  APInt apInt = apValue.bitcastToAPInt();
  apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/true);
  os << str;
}

void ModulePrinter::printLocation(LocationAttr loc) {
  if (printerFlags.shouldPrintDebugInfoPrettyForm()) {
    printLocationInternal(loc, /*pretty=*/true);
  } else {
    os << "loc(";
    printLocationInternal(loc);
    os << ')';
  }
}

/// Returns if the given dialect symbol data is simple enough to print in the
/// pretty form, i.e. without the enclosing "".
static bool isDialectSymbolSimpleEnoughForPrettyForm(StringRef symName) {
  // The name must start with an identifier.
  if (symName.empty() || !isalpha(symName.front()))
    return false;

  // Ignore all the characters that are valid in an identifier in the symbol
  // name.
  symName = symName.drop_while(
      [](char c) { return llvm::isAlnum(c) || c == '.' || c == '_'; });
  if (symName.empty())
    return true;

  // If we got to an unexpected character, then it must be a <>.  Check those
  // recursively.
  if (symName.front() != '<' || symName.back() != '>')
    return false;

  SmallVector<char, 8> nestedPunctuation;
  do {
    // If we ran out of characters, then we had a punctuation mismatch.
    if (symName.empty())
      return false;

    auto c = symName.front();
    symName = symName.drop_front();

    switch (c) {
    // We never allow null characters. This is an EOF indicator for the lexer
    // which we could handle, but isn't important for any known dialect.
    case '\0':
      return false;
    case '<':
    case '[':
    case '(':
    case '{':
      nestedPunctuation.push_back(c);
      continue;
    case '-':
      // Treat `->` as a special token.
      if (!symName.empty() && symName.front() == '>') {
        symName = symName.drop_front();
        continue;
      }
      break;
    // Reject types with mismatched brackets.
    case '>':
      if (nestedPunctuation.pop_back_val() != '<')
        return false;
      break;
    case ']':
      if (nestedPunctuation.pop_back_val() != '[')
        return false;
      break;
    case ')':
      if (nestedPunctuation.pop_back_val() != '(')
        return false;
      break;
    case '}':
      if (nestedPunctuation.pop_back_val() != '{')
        return false;
      break;
    default:
      continue;
    }

    // We're done when the punctuation is fully matched.
  } while (!nestedPunctuation.empty());

  // If there were extra characters, then we failed.
  return symName.empty();
}

/// Print the given dialect symbol to the stream.
static void printDialectSymbol(raw_ostream &os, StringRef symPrefix,
                               StringRef dialectName, StringRef symString) {
  os << symPrefix << dialectName;

  // If this symbol name is simple enough, print it directly in pretty form,
  // otherwise, we print it as an escaped string.
  if (isDialectSymbolSimpleEnoughForPrettyForm(symString)) {
    os << '.' << symString;
    return;
  }

  // TODO: escape the symbol name, it could contain " characters.
  os << "<\"" << symString << "\">";
}

/// Returns if the given string can be represented as a bare identifier.
static bool isBareIdentifier(StringRef name) {
  assert(!name.empty() && "invalid name");

  // By making this unsigned, the value passed in to isalnum will always be
  // in the range 0-255. This is important when building with MSVC because
  // its implementation will assert. This situation can arise when dealing
  // with UTF-8 multibyte characters.
  unsigned char firstChar = static_cast<unsigned char>(name[0]);
  if (!isalpha(firstChar) && firstChar != '_')
    return false;
  return llvm::all_of(name.drop_front(), [](unsigned char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

/// Print the given string as a symbol reference. A symbol reference is
/// represented as a string prefixed with '@'. The reference is surrounded with
/// ""'s and escaped if it has any special or non-printable characters in it.
static void printSymbolReference(StringRef symbolRef, raw_ostream &os) {
  assert(!symbolRef.empty() && "expected valid symbol reference");

  // If the symbol can be represented as a bare identifier, write it directly.
  if (isBareIdentifier(symbolRef)) {
    os << '@' << symbolRef;
    return;
  }

  // Otherwise, output the reference wrapped in quotes with proper escaping.
  os << "@\"";
  printEscapedString(symbolRef, os);
  os << '"';
}

// Print out a valid ElementsAttr that is succinct and can represent any
// potential shape/type, for use when eliding a large ElementsAttr.
//
// We choose to use an opaque ElementsAttr literal with conspicuous content to
// hopefully alert readers to the fact that this has been elided.
//
// Unfortunately, neither of the strings of an opaque ElementsAttr literal will
// accept the string "elided". The first string must be a registered dialect
// name and the latter must be a hex constant.
static void printElidedElementsAttr(raw_ostream &os) {
  os << R"(opaque<"", "0xDEADBEEF">)";
}

void ModulePrinter::printAttribute(Attribute attr,
                                   AttrTypeElision typeElision) {
  if (!attr) {
    os << "<<NULL ATTRIBUTE>>";
    return;
  }

  // Check for an alias for this attribute.
  if (state) {
    Twine alias = state->getAliasState().getAttributeAlias(attr);
    if (!alias.isTriviallyEmpty()) {
      os << '#' << alias;
      return;
    }
  }

  auto attrType = attr.getType();
  switch (attr.getKind()) {
  default:
    return printDialectAttribute(attr);

  case StandardAttributes::Opaque: {
    auto opaqueAttr = attr.cast<OpaqueAttr>();
    printDialectSymbol(os, "#", opaqueAttr.getDialectNamespace(),
                       opaqueAttr.getAttrData());
    break;
  }
  case StandardAttributes::Unit:
    os << "unit";
    break;
  case StandardAttributes::Bool:
    os << (attr.cast<BoolAttr>().getValue() ? "true" : "false");

    // BoolAttr always elides the type.
    return;
  case StandardAttributes::Dictionary:
    os << '{';
    interleaveComma(attr.cast<DictionaryAttr>().getValue(),
                    [&](NamedAttribute attr) { printNamedAttribute(attr); });
    os << '}';
    break;
  case StandardAttributes::Integer: {
    auto intAttr = attr.cast<IntegerAttr>();
    // Print all signed/signless integer attributes as signed unless i1.
    bool isSigned =
        attrType.isIndex() || (!attrType.isUnsignedInteger() &&
                               attrType.getIntOrFloatBitWidth() != 1);
    intAttr.getValue().print(os, isSigned);

    // IntegerAttr elides the type if I64.
    if (typeElision == AttrTypeElision::May && attrType.isSignlessInteger(64))
      return;
    break;
  }
  case StandardAttributes::Float: {
    auto floatAttr = attr.cast<FloatAttr>();
    printFloatValue(floatAttr.getValue(), os);

    // FloatAttr elides the type if F64.
    if (typeElision == AttrTypeElision::May && attrType.isF64())
      return;
    break;
  }
  case StandardAttributes::String:
    os << '"';
    printEscapedString(attr.cast<StringAttr>().getValue(), os);
    os << '"';
    break;
  case StandardAttributes::Array:
    os << '[';
    interleaveComma(attr.cast<ArrayAttr>().getValue(), [&](Attribute attr) {
      printAttribute(attr, AttrTypeElision::May);
    });
    os << ']';
    break;
  case StandardAttributes::AffineMap:
    os << "affine_map<";
    attr.cast<AffineMapAttr>().getValue().print(os);
    os << '>';

    // AffineMap always elides the type.
    return;
  case StandardAttributes::IntegerSet:
    os << "affine_set<";
    attr.cast<IntegerSetAttr>().getValue().print(os);
    os << '>';

    // IntegerSet always elides the type.
    return;
  case StandardAttributes::Type:
    printType(attr.cast<TypeAttr>().getValue());
    break;
  case StandardAttributes::SymbolRef: {
    auto refAttr = attr.dyn_cast<SymbolRefAttr>();
    printSymbolReference(refAttr.getRootReference(), os);
    for (FlatSymbolRefAttr nestedRef : refAttr.getNestedReferences()) {
      os << "::";
      printSymbolReference(nestedRef.getValue(), os);
    }
    break;
  }
  case StandardAttributes::OpaqueElements: {
    auto eltsAttr = attr.cast<OpaqueElementsAttr>();
    if (printerFlags.shouldElideElementsAttr(eltsAttr)) {
      printElidedElementsAttr(os);
      break;
    }
    os << "opaque<\"" << eltsAttr.getDialect()->getNamespace() << "\", ";
    os << '"' << "0x" << llvm::toHex(eltsAttr.getValue()) << "\">";
    break;
  }
  case StandardAttributes::DenseElements: {
    auto eltsAttr = attr.cast<DenseElementsAttr>();
    if (printerFlags.shouldElideElementsAttr(eltsAttr)) {
      printElidedElementsAttr(os);
      break;
    }
    os << "dense<";
    printDenseElementsAttr(eltsAttr, /*allowHex=*/true);
    os << '>';
    break;
  }
  case StandardAttributes::SparseElements: {
    auto elementsAttr = attr.cast<SparseElementsAttr>();
    if (printerFlags.shouldElideElementsAttr(elementsAttr.getIndices()) ||
        printerFlags.shouldElideElementsAttr(elementsAttr.getValues())) {
      printElidedElementsAttr(os);
      break;
    }
    os << "sparse<";
    printDenseElementsAttr(elementsAttr.getIndices(), /*allowHex=*/false);
    os << ", ";
    printDenseElementsAttr(elementsAttr.getValues(), /*allowHex=*/true);
    os << '>';
    break;
  }

  // Location attributes.
  case StandardAttributes::CallSiteLocation:
  case StandardAttributes::FileLineColLocation:
  case StandardAttributes::FusedLocation:
  case StandardAttributes::NameLocation:
  case StandardAttributes::OpaqueLocation:
  case StandardAttributes::UnknownLocation:
    printLocation(attr.cast<LocationAttr>());
    break;
  }

  // Don't print the type if we must elide it, or if it is a None type.
  if (typeElision != AttrTypeElision::Must && !attrType.isa<NoneType>()) {
    os << " : ";
    printType(attrType);
  }
}

/// Print the integer element of the given DenseElementsAttr at 'index'.
static void printDenseIntElement(DenseElementsAttr attr, raw_ostream &os,
                                 unsigned index, bool isSigned) {
  APInt value = *std::next(attr.int_value_begin(), index);
  if (value.getBitWidth() == 1)
    os << (value.getBoolValue() ? "true" : "false");
  else
    value.print(os, isSigned);
}

/// Print the float element of the given DenseElementsAttr at 'index'.
static void printDenseFloatElement(DenseElementsAttr attr, raw_ostream &os,
                                   unsigned index, bool isSigned) {
  assert(isSigned && "floating point values are always signed");
  APFloat value = *std::next(attr.float_value_begin(), index);
  printFloatValue(value, os);
}

void ModulePrinter::printDenseElementsAttr(DenseElementsAttr attr,
                                           bool allowHex) {
  auto type = attr.getType();
  auto shape = type.getShape();
  auto rank = type.getRank();
  bool isSigned = !type.getElementType().isUnsignedInteger();

  // The function used to print elements of this attribute.
  auto printEltFn = type.getElementType().isa<IntegerType>()
                        ? printDenseIntElement
                        : printDenseFloatElement;

  // Special case for 0-d and splat tensors.
  if (attr.isSplat()) {
    printEltFn(attr, os, 0, isSigned);
    return;
  }

  // Special case for degenerate tensors.
  auto numElements = type.getNumElements();
  if (numElements == 0) {
    for (int i = 0; i < rank; ++i)
      os << '[';
    for (int i = 0; i < rank; ++i)
      os << ']';
    return;
  }

  // Check to see if we should format this attribute as a hex string.
  if (allowHex && printElementsAttrWithHexIfLarger != -1 &&
      numElements > printElementsAttrWithHexIfLarger) {
    ArrayRef<char> rawData = attr.getRawData();
    os << '"' << "0x" << llvm::toHex(StringRef(rawData.data(), rawData.size()))
       << "\"";
    return;
  }

  // We use a mixed-radix counter to iterate through the shape. When we bump a
  // non-least-significant digit, we emit a close bracket. When we next emit an
  // element we re-open all closed brackets.

  // The mixed-radix counter, with radices in 'shape'.
  SmallVector<unsigned, 4> counter(rank, 0);
  // The number of brackets that have been opened and not closed.
  unsigned openBrackets = 0;

  auto bumpCounter = [&]() {
    // Bump the least significant digit.
    ++counter[rank - 1];
    // Iterate backwards bubbling back the increment.
    for (unsigned i = rank - 1; i > 0; --i)
      if (counter[i] >= shape[i]) {
        // Index 'i' is rolled over. Bump (i-1) and close a bracket.
        counter[i] = 0;
        ++counter[i - 1];
        --openBrackets;
        os << ']';
      }
  };

  for (unsigned idx = 0, e = numElements; idx != e; ++idx) {
    if (idx != 0)
      os << ", ";
    while (openBrackets++ < rank)
      os << '[';
    openBrackets = rank;
    printEltFn(attr, os, idx, isSigned);
    bumpCounter();
  }
  while (openBrackets-- > 0)
    os << ']';
}

void ModulePrinter::printType(Type type) {
  // Check for an alias for this type.
  if (state) {
    StringRef alias = state->getAliasState().getTypeAlias(type);
    if (!alias.empty()) {
      os << '!' << alias;
      return;
    }
  }

  switch (type.getKind()) {
  default:
    return printDialectType(type);

  case Type::Kind::Opaque: {
    auto opaqueTy = type.cast<OpaqueType>();
    printDialectSymbol(os, "!", opaqueTy.getDialectNamespace(),
                       opaqueTy.getTypeData());
    return;
  }
  case StandardTypes::Index:
    os << "index";
    return;
  case StandardTypes::BF16:
    os << "bf16";
    return;
  case StandardTypes::F16:
    os << "f16";
    return;
  case StandardTypes::F32:
    os << "f32";
    return;
  case StandardTypes::F64:
    os << "f64";
    return;

  case StandardTypes::Integer: {
    auto integer = type.cast<IntegerType>();
    if (integer.isSigned())
      os << 's';
    else if (integer.isUnsigned())
      os << 'u';
    os << 'i' << integer.getWidth();
    return;
  }
  case Type::Kind::Function: {
    auto func = type.cast<FunctionType>();
    os << '(';
    interleaveComma(func.getInputs(), [&](Type type) { printType(type); });
    os << ") -> ";
    auto results = func.getResults();
    if (results.size() == 1 && !results[0].isa<FunctionType>())
      os << results[0];
    else {
      os << '(';
      interleaveComma(results, [&](Type type) { printType(type); });
      os << ')';
    }
    return;
  }
  case StandardTypes::Vector: {
    auto v = type.cast<VectorType>();
    os << "vector<";
    for (auto dim : v.getShape())
      os << dim << 'x';
    os << v.getElementType() << '>';
    return;
  }
  case StandardTypes::RankedTensor: {
    auto v = type.cast<RankedTensorType>();
    os << "tensor<";
    for (auto dim : v.getShape()) {
      if (dim < 0)
        os << '?';
      else
        os << dim;
      os << 'x';
    }
    os << v.getElementType() << '>';
    return;
  }
  case StandardTypes::UnrankedTensor: {
    auto v = type.cast<UnrankedTensorType>();
    os << "tensor<*x";
    printType(v.getElementType());
    os << '>';
    return;
  }
  case StandardTypes::MemRef: {
    auto v = type.cast<MemRefType>();
    os << "memref<";
    for (auto dim : v.getShape()) {
      if (dim < 0)
        os << '?';
      else
        os << dim;
      os << 'x';
    }
    printType(v.getElementType());
    for (auto map : v.getAffineMaps()) {
      os << ", ";
      printAttribute(AffineMapAttr::get(map));
    }
    // Only print the memory space if it is the non-default one.
    if (v.getMemorySpace())
      os << ", " << v.getMemorySpace();
    os << '>';
    return;
  }
  case StandardTypes::UnrankedMemRef: {
    auto v = type.cast<UnrankedMemRefType>();
    os << "memref<*x";
    printType(v.getElementType());
    os << '>';
    return;
  }
  case StandardTypes::Complex:
    os << "complex<";
    printType(type.cast<ComplexType>().getElementType());
    os << '>';
    return;
  case StandardTypes::Tuple: {
    auto tuple = type.cast<TupleType>();
    os << "tuple<";
    interleaveComma(tuple.getTypes(), [&](Type type) { printType(type); });
    os << '>';
    return;
  }
  case StandardTypes::None:
    os << "none";
    return;
  }
}

void ModulePrinter::printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                                          ArrayRef<StringRef> elidedAttrs,
                                          bool withKeyword) {
  // If there are no attributes, then there is nothing to be done.
  if (attrs.empty())
    return;

  // Filter out any attributes that shouldn't be included.
  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
        return !llvm::is_contained(elidedAttrs, attr.first.strref());
      }));

  // If there are no attributes left to print after filtering, then we're done.
  if (filteredAttrs.empty())
    return;

  // Print the 'attributes' keyword if necessary.
  if (withKeyword)
    os << " attributes";

  // Otherwise, print them all out in braces.
  os << " {";
  interleaveComma(filteredAttrs,
                  [&](NamedAttribute attr) { printNamedAttribute(attr); });
  os << '}';
}

void ModulePrinter::printNamedAttribute(NamedAttribute attr) {
  if (isBareIdentifier(attr.first)) {
    os << attr.first;
  } else {
    os << '"';
    printEscapedString(attr.first.strref(), os);
    os << '"';
  }

  // Pretty printing elides the attribute value for unit attributes.
  if (attr.second.isa<UnitAttr>())
    return;

  os << " = ";
  printAttribute(attr.second);
}

//===----------------------------------------------------------------------===//
// CustomDialectAsmPrinter
//===----------------------------------------------------------------------===//

namespace {
/// This class provides the main specialization of the DialectAsmPrinter that is
/// used to provide support for print attributes and types. This hooks allows
/// for dialects to hook into the main ModulePrinter.
struct CustomDialectAsmPrinter : public DialectAsmPrinter {
public:
  CustomDialectAsmPrinter(ModulePrinter &printer) : printer(printer) {}
  ~CustomDialectAsmPrinter() override {}

  raw_ostream &getStream() const override { return printer.getStream(); }

  /// Print the given attribute to the stream.
  void printAttribute(Attribute attr) override { printer.printAttribute(attr); }

  /// Print the given floating point value in a stablized form.
  void printFloat(const APFloat &value) override {
    printFloatValue(value, getStream());
  }

  /// Print the given type to the stream.
  void printType(Type type) override { printer.printType(type); }

  /// The main module printer.
  ModulePrinter &printer;
};
} // end anonymous namespace

void ModulePrinter::printDialectAttribute(Attribute attr) {
  auto &dialect = attr.getDialect();

  // Ask the dialect to serialize the attribute to a string.
  std::string attrName;
  {
    llvm::raw_string_ostream attrNameStr(attrName);
    ModulePrinter subPrinter(attrNameStr, printerFlags, state);
    CustomDialectAsmPrinter printer(subPrinter);
    dialect.printAttribute(attr, printer);
  }
  printDialectSymbol(os, "#", dialect.getNamespace(), attrName);
}

void ModulePrinter::printDialectType(Type type) {
  auto &dialect = type.getDialect();

  // Ask the dialect to serialize the type to a string.
  std::string typeName;
  {
    llvm::raw_string_ostream typeNameStr(typeName);
    ModulePrinter subPrinter(typeNameStr, printerFlags, state);
    CustomDialectAsmPrinter printer(subPrinter);
    dialect.printType(type, printer);
  }
  printDialectSymbol(os, "!", dialect.getNamespace(), typeName);
}

//===----------------------------------------------------------------------===//
// Affine expressions and maps
//===----------------------------------------------------------------------===//

void ModulePrinter::printAffineExpr(
    AffineExpr expr, function_ref<void(unsigned, bool)> printValueName) {
  printAffineExprInternal(expr, BindingStrength::Weak, printValueName);
}

void ModulePrinter::printAffineExprInternal(
    AffineExpr expr, BindingStrength enclosingTightness,
    function_ref<void(unsigned, bool)> printValueName) {
  const char *binopSpelling = nullptr;
  switch (expr.getKind()) {
  case AffineExprKind::SymbolId: {
    unsigned pos = expr.cast<AffineSymbolExpr>().getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/true);
    else
      os << 's' << pos;
    return;
  }
  case AffineExprKind::DimId: {
    unsigned pos = expr.cast<AffineDimExpr>().getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/false);
    else
      os << 'd' << pos;
    return;
  }
  case AffineExprKind::Constant:
    os << expr.cast<AffineConstantExpr>().getValue();
    return;
  case AffineExprKind::Add:
    binopSpelling = " + ";
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::FloorDiv:
    binopSpelling = " floordiv ";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = " ceildiv ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " mod ";
    break;
  }

  auto binOp = expr.cast<AffineBinaryOpExpr>();
  AffineExpr lhsExpr = binOp.getLHS();
  AffineExpr rhsExpr = binOp.getRHS();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (enclosingTightness == BindingStrength::Strong)
      os << '(';

    // Pretty print multiplication with -1.
    auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>();
    if (rhsConst && binOp.getKind() == AffineExprKind::Mul &&
        rhsConst.getValue() == -1) {
      os << "-";
      printAffineExprInternal(lhsExpr, BindingStrength::Strong, printValueName);
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }

    printAffineExprInternal(lhsExpr, BindingStrength::Strong, printValueName);

    os << binopSpelling;
    printAffineExprInternal(rhsExpr, BindingStrength::Strong, printValueName);

    if (enclosingTightness == BindingStrength::Strong)
      os << ')';
    return;
  }

  // Print out special "pretty" forms for add.
  if (enclosingTightness == BindingStrength::Strong)
    os << '(';

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (auto rhs = rhsExpr.dyn_cast<AffineBinaryOpExpr>()) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      AffineExpr rrhsExpr = rhs.getRHS();
      if (auto rrhs = rrhsExpr.dyn_cast<AffineConstantExpr>()) {
        if (rrhs.getValue() == -1) {
          printAffineExprInternal(lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          if (rhs.getLHS().getKind() == AffineExprKind::Add) {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong,
                                    printValueName);
          } else {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Weak,
                                    printValueName);
          }

          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }

        if (rrhs.getValue() < -1) {
          printAffineExprInternal(lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong,
                                  printValueName);
          os << " * " << -rrhs.getValue();
          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>()) {
    if (rhsConst.getValue() < 0) {
      printAffineExprInternal(lhsExpr, BindingStrength::Weak, printValueName);
      os << " - " << -rhsConst.getValue();
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }
  }

  printAffineExprInternal(lhsExpr, BindingStrength::Weak, printValueName);

  os << " + ";
  printAffineExprInternal(rhsExpr, BindingStrength::Weak, printValueName);

  if (enclosingTightness == BindingStrength::Strong)
    os << ')';
}

void ModulePrinter::printAffineConstraint(AffineExpr expr, bool isEq) {
  printAffineExprInternal(expr, BindingStrength::Weak);
  isEq ? os << " == 0" : os << " >= 0";
}

void ModulePrinter::printAffineMap(AffineMap map) {
  // Dimension identifiers.
  os << '(';
  for (int i = 0; i < (int)map.getNumDims() - 1; ++i)
    os << 'd' << i << ", ";
  if (map.getNumDims() >= 1)
    os << 'd' << map.getNumDims() - 1;
  os << ')';

  // Symbolic identifiers.
  if (map.getNumSymbols() != 0) {
    os << '[';
    for (unsigned i = 0; i < map.getNumSymbols() - 1; ++i)
      os << 's' << i << ", ";
    if (map.getNumSymbols() >= 1)
      os << 's' << map.getNumSymbols() - 1;
    os << ']';
  }

  // Result affine expressions.
  os << " -> (";
  interleaveComma(map.getResults(),
                  [&](AffineExpr expr) { printAffineExpr(expr); });
  os << ')';
}

void ModulePrinter::printIntegerSet(IntegerSet set) {
  // Dimension identifiers.
  os << '(';
  for (unsigned i = 1; i < set.getNumDims(); ++i)
    os << 'd' << i - 1 << ", ";
  if (set.getNumDims() >= 1)
    os << 'd' << set.getNumDims() - 1;
  os << ')';

  // Symbolic identifiers.
  if (set.getNumSymbols() != 0) {
    os << '[';
    for (unsigned i = 0; i < set.getNumSymbols() - 1; ++i)
      os << 's' << i << ", ";
    if (set.getNumSymbols() >= 1)
      os << 's' << set.getNumSymbols() - 1;
    os << ']';
  }

  // Print constraints.
  os << " : (";
  int numConstraints = set.getNumConstraints();
  for (int i = 1; i < numConstraints; ++i) {
    printAffineConstraint(set.getConstraint(i - 1), set.isEq(i - 1));
    os << ", ";
  }
  if (numConstraints >= 1)
    printAffineConstraint(set.getConstraint(numConstraints - 1),
                          set.isEq(numConstraints - 1));
  os << ')';
}

//===----------------------------------------------------------------------===//
// OperationPrinter
//===----------------------------------------------------------------------===//

namespace {
/// This class contains the logic for printing operations, regions, and blocks.
class OperationPrinter : public ModulePrinter, private OpAsmPrinter {
public:
  explicit OperationPrinter(raw_ostream &os, OpPrintingFlags flags,
                            AsmStateImpl &state)
      : ModulePrinter(os, flags, &state) {}

  /// Print the given top-level module.
  void print(ModuleOp op);
  /// Print the given operation with its indent and location.
  void print(Operation *op);
  /// Print the bare location, not including indentation/location/etc.
  void printOperation(Operation *op);
  /// Print the given operation in the generic form.
  void printGenericOp(Operation *op) override;

  /// Print the name of the given block.
  void printBlockName(Block *block);

  /// Print the given block. If 'printBlockArgs' is false, the arguments of the
  /// block are not printed. If 'printBlockTerminator' is false, the terminator
  /// operation of the block is not printed.
  void print(Block *block, bool printBlockArgs = true,
             bool printBlockTerminator = true);

  /// Print the ID of the given value, optionally with its result number.
  void printValueID(Value value, bool printResultNo = true) const;

  //===--------------------------------------------------------------------===//
  // OpAsmPrinter methods
  //===--------------------------------------------------------------------===//

  /// Return the current stream of the printer.
  raw_ostream &getStream() const override { return os; }

  /// Print the given type.
  void printType(Type type) override { ModulePrinter::printType(type); }

  /// Print the given attribute.
  void printAttribute(Attribute attr) override {
    ModulePrinter::printAttribute(attr);
  }

  /// Print the given attribute without its type. The corresponding parser must
  /// provide a valid type for the attribute.
  void printAttributeWithoutType(Attribute attr) override {
    ModulePrinter::printAttribute(attr, AttrTypeElision::Must);
  }

  /// Print the ID for the given value.
  void printOperand(Value value) override { printValueID(value); }

  /// Print an optional attribute dictionary with a given set of elided values.
  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<StringRef> elidedAttrs = {}) override {
    ModulePrinter::printOptionalAttrDict(attrs, elidedAttrs);
  }
  void printOptionalAttrDictWithKeyword(
      ArrayRef<NamedAttribute> attrs,
      ArrayRef<StringRef> elidedAttrs = {}) override {
    ModulePrinter::printOptionalAttrDict(attrs, elidedAttrs,
                                         /*withKeyword=*/true);
  }

  /// Print the given successor.
  void printSuccessor(Block *successor) override;

  /// Print an operation successor with the operands used for the block
  /// arguments.
  void printSuccessorAndUseList(Block *successor,
                                ValueRange succOperands) override;

  /// Print the given region.
  void printRegion(Region &region, bool printEntryBlockArgs,
                   bool printBlockTerminators) override;

  /// Renumber the arguments for the specified region to the same names as the
  /// SSA values in namesToUse. This may only be used for IsolatedFromAbove
  /// operations. If any entry in namesToUse is null, the corresponding
  /// argument name is left alone.
  void shadowRegionArgs(Region &region, ValueRange namesToUse) override {
    state->getSSANameState().shadowRegionArgs(region, namesToUse);
  }

  /// Print the given affine map with the smybol and dimension operands printed
  /// inline with the map.
  void printAffineMapOfSSAIds(AffineMapAttr mapAttr,
                              ValueRange operands) override;

  /// Print the given string as a symbol reference.
  void printSymbolName(StringRef symbolRef) override {
    ::printSymbolReference(symbolRef, os);
  }

private:
  /// The number of spaces used for indenting nested operations.
  const static unsigned indentWidth = 2;

  // This is the current indentation level for nested structures.
  unsigned currentIndent = 0;
};
} // end anonymous namespace

void OperationPrinter::print(ModuleOp op) {
  // Output the aliases at the top level.
  state->getAliasState().printAttributeAliases(os, newLine);
  state->getAliasState().printTypeAliases(os, newLine);

  // Print the module.
  print(op.getOperation());
}

void OperationPrinter::print(Operation *op) {
  // Track the location of this operation.
  state->registerOperationLocation(op, newLine.curLine, currentIndent);

  os.indent(currentIndent);
  printOperation(op);
  printTrailingLocation(op->getLoc());
}

void OperationPrinter::printOperation(Operation *op) {
  if (size_t numResults = op->getNumResults()) {
    auto printResultGroup = [&](size_t resultNo, size_t resultCount) {
      printValueID(op->getResult(resultNo), /*printResultNo=*/false);
      if (resultCount > 1)
        os << ':' << resultCount;
    };

    // Check to see if this operation has multiple result groups.
    ArrayRef<int> resultGroups = state->getSSANameState().getOpResultGroups(op);
    if (!resultGroups.empty()) {
      // Interleave the groups excluding the last one, this one will be handled
      // separately.
      interleaveComma(llvm::seq<int>(0, resultGroups.size() - 1), [&](int i) {
        printResultGroup(resultGroups[i],
                         resultGroups[i + 1] - resultGroups[i]);
      });
      os << ", ";
      printResultGroup(resultGroups.back(), numResults - resultGroups.back());

    } else {
      printResultGroup(/*resultNo=*/0, /*resultCount=*/numResults);
    }

    os << " = ";
  }

  // If requested, always print the generic form.
  if (!printerFlags.shouldPrintGenericOpForm()) {
    // Check to see if this is a known operation.  If so, use the registered
    // custom printer hook.
    if (auto *opInfo = op->getAbstractOperation()) {
      opInfo->printAssembly(op, *this);
      return;
    }
  }

  // Otherwise print with the generic assembly form.
  printGenericOp(op);
}

void OperationPrinter::printGenericOp(Operation *op) {
  os << '"';
  printEscapedString(op->getName().getStringRef(), os);
  os << "\"(";
  interleaveComma(op->getOperands(), [&](Value value) { printValueID(value); });
  os << ')';

  // For terminators, print the list of successors and their operands.
  if (op->getNumSuccessors() != 0) {
    os << '[';
    interleaveComma(op->getSuccessors(),
                    [&](Block *successor) { printBlockName(successor); });
    os << ']';
  }

  // Print regions.
  if (op->getNumRegions() != 0) {
    os << " (";
    interleaveComma(op->getRegions(), [&](Region &region) {
      printRegion(region, /*printEntryBlockArgs=*/true,
                  /*printBlockTerminators=*/true);
    });
    os << ')';
  }

  auto attrs = op->getAttrs();
  printOptionalAttrDict(attrs);

  // Print the type signature of the operation.
  os << " : ";
  printFunctionalType(op);
}

void OperationPrinter::printBlockName(Block *block) {
  auto id = state->getSSANameState().getBlockID(block);
  if (id != SSANameState::NameSentinel)
    os << "^bb" << id;
  else
    os << "^INVALIDBLOCK";
}

void OperationPrinter::print(Block *block, bool printBlockArgs,
                             bool printBlockTerminator) {
  // Print the block label and argument list if requested.
  if (printBlockArgs) {
    os.indent(currentIndent);
    printBlockName(block);

    // Print the argument list if non-empty.
    if (!block->args_empty()) {
      os << '(';
      interleaveComma(block->getArguments(), [&](BlockArgument arg) {
        printValueID(arg);
        os << ": ";
        printType(arg.getType());
      });
      os << ')';
    }
    os << ':';

    // Print out some context information about the predecessors of this block.
    if (!block->getParent()) {
      os << "\t// block is not in a region!";
    } else if (block->hasNoPredecessors()) {
      os << "\t// no predecessors";
    } else if (auto *pred = block->getSinglePredecessor()) {
      os << "\t// pred: ";
      printBlockName(pred);
    } else {
      // We want to print the predecessors in increasing numeric order, not in
      // whatever order the use-list is in, so gather and sort them.
      SmallVector<std::pair<unsigned, Block *>, 4> predIDs;
      for (auto *pred : block->getPredecessors())
        predIDs.push_back({state->getSSANameState().getBlockID(pred), pred});
      llvm::array_pod_sort(predIDs.begin(), predIDs.end());

      os << "\t// " << predIDs.size() << " preds: ";

      interleaveComma(predIDs, [&](std::pair<unsigned, Block *> pred) {
        printBlockName(pred.second);
      });
    }
    os << newLine;
  }

  currentIndent += indentWidth;
  auto range = llvm::make_range(
      block->getOperations().begin(),
      std::prev(block->getOperations().end(), printBlockTerminator ? 0 : 1));
  for (auto &op : range) {
    print(&op);
    os << newLine;
  }
  currentIndent -= indentWidth;
}

void OperationPrinter::printValueID(Value value, bool printResultNo) const {
  state->getSSANameState().printValueID(value, printResultNo, os);
}

void OperationPrinter::printSuccessor(Block *successor) {
  printBlockName(successor);
}

void OperationPrinter::printSuccessorAndUseList(Block *successor,
                                                ValueRange succOperands) {
  printBlockName(successor);
  if (succOperands.empty())
    return;

  os << '(';
  interleaveComma(succOperands,
                  [this](Value operand) { printValueID(operand); });
  os << " : ";
  interleaveComma(succOperands,
                  [this](Value operand) { printType(operand.getType()); });
  os << ')';
}

void OperationPrinter::printRegion(Region &region, bool printEntryBlockArgs,
                                   bool printBlockTerminators) {
  os << " {" << newLine;
  if (!region.empty()) {
    auto *entryBlock = &region.front();
    print(entryBlock, printEntryBlockArgs && entryBlock->getNumArguments() != 0,
          printBlockTerminators);
    for (auto &b : llvm::drop_begin(region.getBlocks(), 1))
      print(&b);
  }
  os.indent(currentIndent) << "}";
}

void OperationPrinter::printAffineMapOfSSAIds(AffineMapAttr mapAttr,
                                              ValueRange operands) {
  AffineMap map = mapAttr.getValue();
  unsigned numDims = map.getNumDims();
  auto printValueName = [&](unsigned pos, bool isSymbol) {
    unsigned index = isSymbol ? numDims + pos : pos;
    assert(index < operands.size());
    if (isSymbol)
      os << "symbol(";
    printValueID(operands[index]);
    if (isSymbol)
      os << ')';
  };

  interleaveComma(map.getResults(), [&](AffineExpr expr) {
    printAffineExpr(expr, printValueName);
  });
}

//===----------------------------------------------------------------------===//
// print and dump methods
//===----------------------------------------------------------------------===//

void Attribute::print(raw_ostream &os) const {
  ModulePrinter(os).printAttribute(*this);
}

void Attribute::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void Type::print(raw_ostream &os) { ModulePrinter(os).printType(*this); }

void Type::dump() { print(llvm::errs()); }

void AffineMap::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void IntegerSet::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineExpr::print(raw_ostream &os) const {
  if (expr == nullptr) {
    os << "null affine expr";
    return;
  }
  ModulePrinter(os).printAffineExpr(*this);
}

void AffineExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineMap::print(raw_ostream &os) const {
  if (map == nullptr) {
    os << "null affine map";
    return;
  }
  ModulePrinter(os).printAffineMap(*this);
}

void IntegerSet::print(raw_ostream &os) const {
  ModulePrinter(os).printIntegerSet(*this);
}

void Value::print(raw_ostream &os) {
  if (auto *op = getDefiningOp())
    return op->print(os);
  // TODO: Improve this.
  assert(isa<BlockArgument>());
  os << "<block argument>\n";
}
void Value::print(raw_ostream &os, AsmState &state) {
  if (auto *op = getDefiningOp())
    return op->print(os, state);

  // TODO: Improve this.
  assert(isa<BlockArgument>());
  os << "<block argument>\n";
}

void Value::dump() {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void Value::printAsOperand(raw_ostream &os, AsmState &state) {
  // TODO(riverriddle) This doesn't necessarily capture all potential cases.
  // Currently, region arguments can be shadowed when printing the main
  // operation. If the IR hasn't been printed, this will produce the old SSA
  // name and not the shadowed name.
  state.getImpl().getSSANameState().printValueID(*this, /*printResultNo=*/true,
                                                 os);
}

void Operation::print(raw_ostream &os, OpPrintingFlags flags) {
  // Handle top-level operations or local printing.
  if (!getParent() || flags.shouldUseLocalScope()) {
    AsmState state(this);
    OperationPrinter(os, flags, state.getImpl()).print(this);
    return;
  }

  Operation *parentOp = getParentOp();
  if (!parentOp) {
    os << "<<UNLINKED OPERATION>>\n";
    return;
  }
  // Get the top-level op.
  while (auto *nextOp = parentOp->getParentOp())
    parentOp = nextOp;

  AsmState state(parentOp);
  print(os, state, flags);
}
void Operation::print(raw_ostream &os, AsmState &state, OpPrintingFlags flags) {
  OperationPrinter(os, flags, state.getImpl()).print(this);
}

void Operation::dump() {
  print(llvm::errs(), OpPrintingFlags().useLocalScope());
  llvm::errs() << "\n";
}

void Block::print(raw_ostream &os) {
  Operation *parentOp = getParentOp();
  if (!parentOp) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }
  // Get the top-level op.
  while (auto *nextOp = parentOp->getParentOp())
    parentOp = nextOp;

  AsmState state(parentOp);
  print(os, state);
}
void Block::print(raw_ostream &os, AsmState &state) {
  OperationPrinter(os, /*flags=*/llvm::None, state.getImpl()).print(this);
}

void Block::dump() { print(llvm::errs()); }

/// Print out the name of the block without printing its body.
void Block::printAsOperand(raw_ostream &os, bool printType) {
  Operation *parentOp = getParentOp();
  if (!parentOp) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }
  // Get the top-level op.
  while (auto *nextOp = parentOp->getParentOp())
    parentOp = nextOp;

  AsmState state(parentOp);
  printAsOperand(os, state);
}
void Block::printAsOperand(raw_ostream &os, AsmState &state) {
  OperationPrinter printer(os, /*flags=*/llvm::None, state.getImpl());
  printer.printBlockName(this);
}

void ModuleOp::print(raw_ostream &os, OpPrintingFlags flags) {
  AsmState state(*this);

  // Don't populate aliases when printing at local scope.
  if (!flags.shouldUseLocalScope())
    state.getImpl().initializeAliases(*this);
  print(os, state, flags);
}
void ModuleOp::print(raw_ostream &os, AsmState &state, OpPrintingFlags flags) {
  OperationPrinter(os, flags, state.getImpl()).print(*this);
}

void ModuleOp::dump() { print(llvm::errs()); }
