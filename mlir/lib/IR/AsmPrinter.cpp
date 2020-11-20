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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
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

namespace {
/// This struct contains command line options that can be used to initialize
/// various bits of the AsmPrinter. This uses a struct wrapper to avoid the need
/// for global command line options.
struct AsmPrinterOptions {
  llvm::cl::opt<int64_t> printElementsAttrWithHexIfLarger{
      "mlir-print-elementsattrs-with-hex-if-larger",
      llvm::cl::desc(
          "Print DenseElementsAttrs with a hex string that have "
          "more elements than the given upper limit (use -1 to disable)")};

  llvm::cl::opt<unsigned> elideElementsAttrIfLarger{
      "mlir-elide-elementsattrs-if-larger",
      llvm::cl::desc("Elide ElementsAttrs with \"...\" that have "
                     "more elements than the given upper limit")};

  llvm::cl::opt<bool> printDebugInfoOpt{
      "mlir-print-debuginfo", llvm::cl::init(false),
      llvm::cl::desc("Print debug info in MLIR output")};

  llvm::cl::opt<bool> printPrettyDebugInfoOpt{
      "mlir-pretty-debuginfo", llvm::cl::init(false),
      llvm::cl::desc("Print pretty debug info in MLIR output")};

  // Use the generic op output form in the operation printer even if the custom
  // form is defined.
  llvm::cl::opt<bool> printGenericOpFormOpt{
      "mlir-print-op-generic", llvm::cl::init(false),
      llvm::cl::desc("Print the generic op form"), llvm::cl::Hidden};

  llvm::cl::opt<bool> printLocalScopeOpt{
      "mlir-print-local-scope", llvm::cl::init(false),
      llvm::cl::desc("Print assuming in local scope by default"),
      llvm::cl::Hidden};
};
} // end anonymous namespace

static llvm::ManagedStatic<AsmPrinterOptions> clOptions;

/// Register a set of useful command-line options that can be used to configure
/// various flags within the AsmPrinter.
void mlir::registerAsmPrinterCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptions;
}

/// Initialize the printing flags with default supplied by the cl::opts above.
OpPrintingFlags::OpPrintingFlags()
    : printDebugInfoFlag(false), printDebugInfoPrettyFormFlag(false),
      printGenericOpFormFlag(false), printLocalScope(false) {
  // Initialize based upon command line options, if they are available.
  if (!clOptions.isConstructed())
    return;
  if (clOptions->elideElementsAttrIfLarger.getNumOccurrences())
    elementsAttrElementLimit = clOptions->elideElementsAttrIfLarger;
  printDebugInfoFlag = clOptions->printDebugInfoOpt;
  printDebugInfoPrettyFormFlag = clOptions->printPrettyDebugInfoOpt;
  printGenericOpFormFlag = clOptions->printGenericOpFormOpt;
  printLocalScope = clOptions->printLocalScopeOpt;
}

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

/// Return the size limit for printing large ElementsAttr.
Optional<int64_t> OpPrintingFlags::getLargeElementsAttrLimit() const {
  return elementsAttrElementLimit;
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

/// Returns true if an ElementsAttr with the given number of elements should be
/// printed with hex.
static bool shouldPrintElementsAttrWithHex(int64_t numElements) {
  // Check to see if a command line option was provided for the limit.
  if (clOptions.isConstructed()) {
    if (clOptions->printElementsAttrWithHexIfLarger.getNumOccurrences()) {
      // -1 is used to disable hex printing.
      if (clOptions->printElementsAttrWithHexIfLarger == -1)
        return false;
      return numElements > clOptions->printElementsAttrWithHexIfLarger;
    }
  }

  // Otherwise, default to printing with hex if the number of elements is >100.
  return numElements > 100;
}

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
// AliasInitializer
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a specific instance of a symbol Alias.
class SymbolAlias {
public:
  SymbolAlias(StringRef name, bool isDeferrable)
      : name(name), suffixIndex(0), hasSuffixIndex(false),
        isDeferrable(isDeferrable) {}
  SymbolAlias(StringRef name, uint32_t suffixIndex, bool isDeferrable)
      : name(name), suffixIndex(suffixIndex), hasSuffixIndex(true),
        isDeferrable(isDeferrable) {}

  /// Print this alias to the given stream.
  void print(raw_ostream &os) const {
    os << name;
    if (hasSuffixIndex)
      os << suffixIndex;
  }

  /// Returns true if this alias supports deferred resolution when parsing.
  bool canBeDeferred() const { return isDeferrable; }

private:
  /// The main name of the alias.
  StringRef name;
  /// The optional suffix index of the alias, if multiple aliases had the same
  /// name.
  uint32_t suffixIndex : 30;
  /// A flag indicating whether this alias has a suffix or not.
  bool hasSuffixIndex : 1;
  /// A flag indicating whether this alias may be deferred or not.
  bool isDeferrable : 1;
};

/// This class represents a utility that initializes the set of attribute and
/// type aliases, without the need to store the extra information within the
/// main AliasState class or pass it around via function arguments.
class AliasInitializer {
public:
  AliasInitializer(
      DialectInterfaceCollection<OpAsmDialectInterface> &interfaces,
      llvm::BumpPtrAllocator &aliasAllocator)
      : interfaces(interfaces), aliasAllocator(aliasAllocator),
        aliasOS(aliasBuffer) {}

  void initialize(Operation *op, const OpPrintingFlags &printerFlags,
                  llvm::MapVector<Attribute, SymbolAlias> &attrToAlias,
                  llvm::MapVector<Type, SymbolAlias> &typeToAlias);

  /// Visit the given attribute to see if it has an alias. `canBeDeferred` is
  /// set to true if the originator of this attribute can resolve the alias
  /// after parsing has completed (e.g. in the case of operation locations).
  void visit(Attribute attr, bool canBeDeferred = false);

  /// Visit the given type to see if it has an alias.
  void visit(Type type);

private:
  /// Try to generate an alias for the provided symbol. If an alias is
  /// generated, the provided alias mapping and reverse mapping are updated.
  /// Returns success if an alias was generated, failure otherwise.
  template <typename T>
  LogicalResult
  generateAlias(T symbol,
                llvm::MapVector<StringRef, std::vector<T>> &aliasToSymbol);

  /// The set of asm interfaces within the context.
  DialectInterfaceCollection<OpAsmDialectInterface> &interfaces;

  /// Mapping between an alias and the set of symbols mapped to it.
  llvm::MapVector<StringRef, std::vector<Attribute>> aliasToAttr;
  llvm::MapVector<StringRef, std::vector<Type>> aliasToType;

  /// An allocator used for alias names.
  llvm::BumpPtrAllocator &aliasAllocator;

  /// The set of visited attributes.
  DenseSet<Attribute> visitedAttributes;

  /// The set of attributes that have aliases *and* can be deferred.
  DenseSet<Attribute> deferrableAttributes;

  /// The set of visited types.
  DenseSet<Type> visitedTypes;

  /// Storage and stream used when generating an alias.
  SmallString<32> aliasBuffer;
  llvm::raw_svector_ostream aliasOS;
};

/// This class implements a dummy OpAsmPrinter that doesn't print any output,
/// and merely collects the attributes and types that *would* be printed in a
/// normal print invocation so that we can generate proper aliases. This allows
/// for us to generate aliases only for the attributes and types that would be
/// in the output, and trims down unnecessary output.
class DummyAliasOperationPrinter : private OpAsmPrinter {
public:
  explicit DummyAliasOperationPrinter(const OpPrintingFlags &flags,
                                      AliasInitializer &initializer)
      : printerFlags(flags), initializer(initializer) {}

  /// Print the given operation.
  void print(Operation *op) {
    // Visit the operation location.
    if (printerFlags.shouldPrintDebugInfo())
      initializer.visit(op->getLoc(), /*canBeDeferred=*/true);

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

private:
  /// Print the given operation in the generic form.
  void printGenericOp(Operation *op) override {
    // Consider nested opertions for aliases.
    if (op->getNumRegions() != 0) {
      for (Region &region : op->getRegions())
        printRegion(region, /*printEntryBlockArgs=*/true,
                    /*printBlockTerminators=*/true);
    }

    // Visit all the types used in the operation.
    for (Type type : op->getOperandTypes())
      printType(type);
    for (Type type : op->getResultTypes())
      printType(type);

    // Consider the attributes of the operation for aliases.
    for (const NamedAttribute &attr : op->getAttrs())
      printAttribute(attr.second);
  }

  /// Print the given block. If 'printBlockArgs' is false, the arguments of the
  /// block are not printed. If 'printBlockTerminator' is false, the terminator
  /// operation of the block is not printed.
  void print(Block *block, bool printBlockArgs = true,
             bool printBlockTerminator = true) {
    // Consider the types of the block arguments for aliases if 'printBlockArgs'
    // is set to true.
    if (printBlockArgs) {
      for (Type type : block->getArgumentTypes())
        printType(type);
    }

    // Consider the operations within this block, ignoring the terminator if
    // requested.
    auto range = llvm::make_range(
        block->begin(), std::prev(block->end(), printBlockTerminator ? 0 : 1));
    for (Operation &op : range)
      print(&op);
  }

  /// Print the given region.
  void printRegion(Region &region, bool printEntryBlockArgs,
                   bool printBlockTerminators) override {
    if (region.empty())
      return;

    auto *entryBlock = &region.front();
    print(entryBlock, printEntryBlockArgs, printBlockTerminators);
    for (Block &b : llvm::drop_begin(region, 1))
      print(&b);
  }

  /// Consider the given type to be printed for an alias.
  void printType(Type type) override { initializer.visit(type); }

  /// Consider the given attribute to be printed for an alias.
  void printAttribute(Attribute attr) override { initializer.visit(attr); }
  void printAttributeWithoutType(Attribute attr) override {
    printAttribute(attr);
  }

  /// Print the given set of attributes with names not included within
  /// 'elidedAttrs'.
  void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                             ArrayRef<StringRef> elidedAttrs = {}) override {
    // Filter out any attributes that shouldn't be included.
    SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
          return !llvm::is_contained(elidedAttrs, attr.first.strref());
        }));
    for (const NamedAttribute &attr : filteredAttrs)
      printAttribute(attr.second);
  }
  void printOptionalAttrDictWithKeyword(
      ArrayRef<NamedAttribute> attrs,
      ArrayRef<StringRef> elidedAttrs = {}) override {
    printOptionalAttrDict(attrs, elidedAttrs);
  }

  /// Return 'nulls' as the output stream, this will ignore any data fed to it.
  raw_ostream &getStream() const override { return llvm::nulls(); }

  /// The following are hooks of `OpAsmPrinter` that are not necessary for
  /// determining potential aliases.
  void printAffineMapOfSSAIds(AffineMapAttr, ValueRange) override {}
  void printOperand(Value) override {}
  void printOperand(Value, raw_ostream &os) override {
    // Users expect the output string to have at least the prefixed % to signal
    // a value name. To maintain this invariant, emit a name even if it is
    // guaranteed to go unused.
    os << "%";
  }
  void printSymbolName(StringRef) override {}
  void printSuccessor(Block *) override {}
  void printSuccessorAndUseList(Block *, ValueRange) override {}
  void shadowRegionArgs(Region &, ValueRange) override {}

  /// The printer flags to use when determining potential aliases.
  const OpPrintingFlags &printerFlags;

  /// The initializer to use when identifying aliases.
  AliasInitializer &initializer;
};
} // end anonymous namespace

/// Sanitize the given name such that it can be used as a valid identifier. If
/// the string needs to be modified in any way, the provided buffer is used to
/// store the new copy,
static StringRef sanitizeIdentifier(StringRef name, SmallString<16> &buffer,
                                    StringRef allowedPunctChars = "$._-",
                                    bool allowTrailingDigit = true) {
  assert(!name.empty() && "Shouldn't have an empty name here");

  auto copyNameToBuffer = [&] {
    for (char ch : name) {
      if (llvm::isAlnum(ch) || allowedPunctChars.contains(ch))
        buffer.push_back(ch);
      else if (ch == ' ')
        buffer.push_back('_');
      else
        buffer.append(llvm::utohexstr((unsigned char)ch));
    }
  };

  // Check to see if this name is valid. If it starts with a digit, then it
  // could conflict with the autogenerated numeric ID's, so add an underscore
  // prefix to avoid problems.
  if (isdigit(name[0])) {
    buffer.push_back('_');
    copyNameToBuffer();
    return buffer;
  }

  // If the name ends with a trailing digit, add a '_' to avoid potential
  // conflicts with autogenerated ID's.
  if (!allowTrailingDigit && isdigit(name.back())) {
    copyNameToBuffer();
    buffer.push_back('_');
    return buffer;
  }

  // Check to see that the name consists of only valid identifier characters.
  for (char ch : name) {
    if (!llvm::isAlnum(ch) && !allowedPunctChars.contains(ch)) {
      copyNameToBuffer();
      return buffer;
    }
  }

  // If there are no invalid characters, return the original name.
  return name;
}

/// Given a collection of aliases and symbols, initialize a mapping from a
/// symbol to a given alias.
template <typename T>
static void
initializeAliases(llvm::MapVector<StringRef, std::vector<T>> &aliasToSymbol,
                  llvm::MapVector<T, SymbolAlias> &symbolToAlias,
                  DenseSet<T> *deferrableAliases = nullptr) {
  std::vector<std::pair<StringRef, std::vector<T>>> aliases =
      aliasToSymbol.takeVector();
  llvm::array_pod_sort(aliases.begin(), aliases.end(),
                       [](const auto *lhs, const auto *rhs) {
                         return lhs->first.compare(rhs->first);
                       });

  for (auto &it : aliases) {
    // If there is only one instance for this alias, use the name directly.
    if (it.second.size() == 1) {
      T symbol = it.second.front();
      bool isDeferrable = deferrableAliases && deferrableAliases->count(symbol);
      symbolToAlias.insert({symbol, SymbolAlias(it.first, isDeferrable)});
      continue;
    }
    // Otherwise, add the index to the name.
    for (int i = 0, e = it.second.size(); i < e; ++i) {
      T symbol = it.second[i];
      bool isDeferrable = deferrableAliases && deferrableAliases->count(symbol);
      symbolToAlias.insert({symbol, SymbolAlias(it.first, i, isDeferrable)});
    }
  }
}

void AliasInitializer::initialize(
    Operation *op, const OpPrintingFlags &printerFlags,
    llvm::MapVector<Attribute, SymbolAlias> &attrToAlias,
    llvm::MapVector<Type, SymbolAlias> &typeToAlias) {
  // Use a dummy printer when walking the IR so that we can collect the
  // attributes/types that will actually be used during printing when
  // considering aliases.
  DummyAliasOperationPrinter aliasPrinter(printerFlags, *this);
  aliasPrinter.print(op);

  // Initialize the aliases sorted by name.
  initializeAliases(aliasToAttr, attrToAlias, &deferrableAttributes);
  initializeAliases(aliasToType, typeToAlias);
}

void AliasInitializer::visit(Attribute attr, bool canBeDeferred) {
  if (!visitedAttributes.insert(attr).second) {
    // If this attribute already has an alias and this instance can't be
    // deferred, make sure that the alias isn't deferred.
    if (!canBeDeferred)
      deferrableAttributes.erase(attr);
    return;
  }

  // Try to generate an alias for this attribute.
  if (succeeded(generateAlias(attr, aliasToAttr))) {
    if (canBeDeferred)
      deferrableAttributes.insert(attr);
    return;
  }

  if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
    for (Attribute element : arrayAttr.getValue())
      visit(element);
  } else if (auto dictAttr = attr.dyn_cast<DictionaryAttr>()) {
    for (const NamedAttribute &attr : dictAttr)
      visit(attr.second);
  } else if (auto typeAttr = attr.dyn_cast<TypeAttr>()) {
    visit(typeAttr.getValue());
  }
}

void AliasInitializer::visit(Type type) {
  if (!visitedTypes.insert(type).second)
    return;

  // Try to generate an alias for this type.
  if (succeeded(generateAlias(type, aliasToType)))
    return;

  // Visit several subtypes that contain types or atttributes.
  if (auto funcType = type.dyn_cast<FunctionType>()) {
    // Visit input and result types for functions.
    for (auto input : funcType.getInputs())
      visit(input);
    for (auto result : funcType.getResults())
      visit(result);
  } else if (auto shapedType = type.dyn_cast<ShapedType>()) {
    visit(shapedType.getElementType());

    // Visit affine maps in memref type.
    if (auto memref = type.dyn_cast<MemRefType>())
      for (auto map : memref.getAffineMaps())
        visit(AffineMapAttr::get(map));
  }
}

template <typename T>
LogicalResult AliasInitializer::generateAlias(
    T symbol, llvm::MapVector<StringRef, std::vector<T>> &aliasToSymbol) {
  SmallString<16> tempBuffer;
  for (const auto &interface : interfaces) {
    interface.getAlias(symbol, aliasOS);
    StringRef name = aliasOS.str();
    if (name.empty())
      continue;
    name = sanitizeIdentifier(name, tempBuffer, /*allowedPunctChars=*/"$_-",
                              /*allowTrailingDigit=*/false);
    name = name.copy(aliasAllocator);

    aliasToSymbol[name].push_back(symbol);
    aliasBuffer.clear();
    return success();
  }
  return failure();
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
  initialize(Operation *op, const OpPrintingFlags &printerFlags,
             DialectInterfaceCollection<OpAsmDialectInterface> &interfaces);

  /// Get an alias for the given attribute if it has one and print it in `os`.
  /// Returns success if an alias was printed, failure otherwise.
  LogicalResult getAlias(Attribute attr, raw_ostream &os) const;

  /// Get an alias for the given type if it has one and print it in `os`.
  /// Returns success if an alias was printed, failure otherwise.
  LogicalResult getAlias(Type ty, raw_ostream &os) const;

  /// Print all of the referenced aliases that can not be resolved in a deferred
  /// manner.
  void printNonDeferredAliases(raw_ostream &os, NewLineCounter &newLine) const {
    printAliases(os, newLine, /*isDeferred=*/false);
  }

  /// Print all of the referenced aliases that support deferred resolution.
  void printDeferredAliases(raw_ostream &os, NewLineCounter &newLine) const {
    printAliases(os, newLine, /*isDeferred=*/true);
  }

private:
  /// Print all of the referenced aliases that support the provided resolution
  /// behavior.
  void printAliases(raw_ostream &os, NewLineCounter &newLine,
                    bool isDeferred) const;

  /// Mapping between attribute and alias.
  llvm::MapVector<Attribute, SymbolAlias> attrToAlias;
  /// Mapping between type and alias.
  llvm::MapVector<Type, SymbolAlias> typeToAlias;

  /// An allocator used for alias names.
  llvm::BumpPtrAllocator aliasAllocator;
};
} // end anonymous namespace

void AliasState::initialize(
    Operation *op, const OpPrintingFlags &printerFlags,
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces) {
  AliasInitializer initializer(interfaces, aliasAllocator);
  initializer.initialize(op, printerFlags, attrToAlias, typeToAlias);
}

LogicalResult AliasState::getAlias(Attribute attr, raw_ostream &os) const {
  auto it = attrToAlias.find(attr);
  if (it == attrToAlias.end())
    return failure();
  it->second.print(os << '#');
  return success();
}

LogicalResult AliasState::getAlias(Type ty, raw_ostream &os) const {
  auto it = typeToAlias.find(ty);
  if (it == typeToAlias.end())
    return failure();

  it->second.print(os << '!');
  return success();
}

void AliasState::printAliases(raw_ostream &os, NewLineCounter &newLine,
                              bool isDeferred) const {
  auto filterFn = [=](const auto &aliasIt) {
    return aliasIt.second.canBeDeferred() == isDeferred;
  };
  for (const auto &it : llvm::make_filter_range(attrToAlias, filterFn)) {
    it.second.print(os << '#');
    os << " = " << it.first << newLine;
  }
  for (const auto &it : llvm::make_filter_range(typeToAlias, filterFn)) {
    it.second.print(os << '!');
    os << " = " << it.first << newLine;
  }
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
  assert(region.getNumArguments() == namesToUse.size() &&
         "incorrect number of names passed in");
  assert(region.getParentOp()->isKnownIsolatedFromAbove() &&
         "only KnownIsolatedFromAbove ops can shadow names");

  SmallVector<char, 16> nameStr;
  for (unsigned i = 0, e = namesToUse.size(); i != e; ++i) {
    auto nameToUse = namesToUse[i];
    if (nameToUse == nullptr)
      continue;
    auto nameToReplace = region.getArgument(i);

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

StringRef SSANameState::uniqueValueName(StringRef name) {
  SmallString<16> tmpBuffer;
  name = sanitizeIdentifier(name, tmpBuffer);

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
      probeName += llvm::utostr(nextConflictID++);
      if (!usedNames.count(probeName)) {
        name = StringRef(probeName).copy(usedNameAllocator);
        break;
      }
      probeName.resize(name.size() + 1);
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
  void initializeAliases(Operation *op, const OpPrintingFlags &printerFlags) {
    aliasState.initialize(op, printerFlags, interfaces);
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
    llvm::interleaveComma(c, os, each_fn);
  }

  /// This enum describes the different kinds of elision for the type of an
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

  /// Print the given location to the stream. If `allowAlias` is true, this
  /// allows for the internal location to use an attribute alias.
  void printLocation(LocationAttr loc, bool allowAlias = false);

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

  /// Print a dense string elements attribute.
  void printDenseStringElementsAttr(DenseStringElementsAttr attr);

  /// Print a dense elements attribute. If 'allowHex' is true, a hex string is
  /// used instead of individual elements when the elements attr is large.
  void printDenseIntOrFPElementsAttr(DenseIntOrFPElementsAttr attr,
                                     bool allowHex);

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
  printLocation(loc, /*allowAlias=*/true);
}

void ModulePrinter::printLocationInternal(LocationAttr loc, bool pretty) {
  TypeSwitch<LocationAttr>(loc)
      .Case<OpaqueLoc>([&](OpaqueLoc loc) {
        printLocationInternal(loc.getFallbackLocation(), pretty);
      })
      .Case<UnknownLoc>([&](UnknownLoc loc) {
        if (pretty)
          os << "[unknown]";
        else
          os << "unknown";
      })
      .Case<FileLineColLoc>([&](FileLineColLoc loc) {
        StringRef mayQuote = pretty ? "" : "\"";
        os << mayQuote << loc.getFilename() << mayQuote << ':' << loc.getLine()
           << ':' << loc.getColumn();
      })
      .Case<NameLoc>([&](NameLoc loc) {
        os << '\"' << loc.getName() << '\"';

        // Print the child if it isn't unknown.
        auto childLoc = loc.getChildLoc();
        if (!childLoc.isa<UnknownLoc>()) {
          os << '(';
          printLocationInternal(childLoc, pretty);
          os << ')';
        }
      })
      .Case<CallSiteLoc>([&](CallSiteLoc loc) {
        Location caller = loc.getCaller();
        Location callee = loc.getCallee();
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
      })
      .Case<FusedLoc>([&](FusedLoc loc) {
        if (!pretty)
          os << "fused";
        if (Attribute metadata = loc.getMetadata())
          os << '<' << metadata << '>';
        os << '[';
        interleave(
            loc.getLocations(),
            [&](Location loc) { printLocationInternal(loc, pretty); },
            [&]() { os << ", "; });
        os << ']';
      });
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

void ModulePrinter::printLocation(LocationAttr loc, bool allowAlias) {
  if (printerFlags.shouldPrintDebugInfoPrettyForm())
    return printLocationInternal(loc, /*pretty=*/true);

  os << "loc(";
  if (!allowAlias || !state || failed(state->getAliasState().getAlias(loc, os)))
    printLocationInternal(loc);
  os << ')';
}

/// Returns true if the given dialect symbol data is simple enough to print in
/// the pretty form, i.e. without the enclosing "".
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

/// Returns true if the given string can be represented as a bare identifier.
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

  // Try to print an alias for this attribute.
  if (state && succeeded(state->getAliasState().getAlias(attr, os)))
    return;

  auto attrType = attr.getType();
  if (auto opaqueAttr = attr.dyn_cast<OpaqueAttr>()) {
    printDialectSymbol(os, "#", opaqueAttr.getDialectNamespace(),
                       opaqueAttr.getAttrData());
  } else if (attr.isa<UnitAttr>()) {
    os << "unit";
    return;
  } else if (auto dictAttr = attr.dyn_cast<DictionaryAttr>()) {
    os << '{';
    interleaveComma(dictAttr.getValue(),
                    [&](NamedAttribute attr) { printNamedAttribute(attr); });
    os << '}';

  } else if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
    if (attrType.isSignlessInteger(1)) {
      os << (intAttr.getValue().getBoolValue() ? "true" : "false");

      // Boolean integer attributes always elides the type.
      return;
    }

    // Only print attributes as unsigned if they are explicitly unsigned or are
    // signless 1-bit values.  Indexes, signed values, and multi-bit signless
    // values print as signed.
    bool isUnsigned =
        attrType.isUnsignedInteger() || attrType.isSignlessInteger(1);
    intAttr.getValue().print(os, !isUnsigned);

    // IntegerAttr elides the type if I64.
    if (typeElision == AttrTypeElision::May && attrType.isSignlessInteger(64))
      return;

  } else if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
    printFloatValue(floatAttr.getValue(), os);

    // FloatAttr elides the type if F64.
    if (typeElision == AttrTypeElision::May && attrType.isF64())
      return;

  } else if (auto strAttr = attr.dyn_cast<StringAttr>()) {
    os << '"';
    printEscapedString(strAttr.getValue(), os);
    os << '"';

  } else if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
    os << '[';
    interleaveComma(arrayAttr.getValue(), [&](Attribute attr) {
      printAttribute(attr, AttrTypeElision::May);
    });
    os << ']';

  } else if (auto affineMapAttr = attr.dyn_cast<AffineMapAttr>()) {
    os << "affine_map<";
    affineMapAttr.getValue().print(os);
    os << '>';

    // AffineMap always elides the type.
    return;

  } else if (auto integerSetAttr = attr.dyn_cast<IntegerSetAttr>()) {
    os << "affine_set<";
    integerSetAttr.getValue().print(os);
    os << '>';

    // IntegerSet always elides the type.
    return;

  } else if (auto typeAttr = attr.dyn_cast<TypeAttr>()) {
    printType(typeAttr.getValue());

  } else if (auto refAttr = attr.dyn_cast<SymbolRefAttr>()) {
    printSymbolReference(refAttr.getRootReference(), os);
    for (FlatSymbolRefAttr nestedRef : refAttr.getNestedReferences()) {
      os << "::";
      printSymbolReference(nestedRef.getValue(), os);
    }

  } else if (auto opaqueAttr = attr.dyn_cast<OpaqueElementsAttr>()) {
    if (printerFlags.shouldElideElementsAttr(opaqueAttr)) {
      printElidedElementsAttr(os);
    } else {
      os << "opaque<\"" << opaqueAttr.getDialect()->getNamespace() << "\", ";
      os << '"' << "0x" << llvm::toHex(opaqueAttr.getValue()) << "\">";
    }

  } else if (auto intOrFpEltAttr = attr.dyn_cast<DenseIntOrFPElementsAttr>()) {
    if (printerFlags.shouldElideElementsAttr(intOrFpEltAttr)) {
      printElidedElementsAttr(os);
    } else {
      os << "dense<";
      printDenseIntOrFPElementsAttr(intOrFpEltAttr, /*allowHex=*/true);
      os << '>';
    }

  } else if (auto strEltAttr = attr.dyn_cast<DenseStringElementsAttr>()) {
    if (printerFlags.shouldElideElementsAttr(strEltAttr)) {
      printElidedElementsAttr(os);
    } else {
      os << "dense<";
      printDenseStringElementsAttr(strEltAttr);
      os << '>';
    }

  } else if (auto sparseEltAttr = attr.dyn_cast<SparseElementsAttr>()) {
    if (printerFlags.shouldElideElementsAttr(sparseEltAttr.getIndices()) ||
        printerFlags.shouldElideElementsAttr(sparseEltAttr.getValues())) {
      printElidedElementsAttr(os);
    } else {
      os << "sparse<";
      DenseIntElementsAttr indices = sparseEltAttr.getIndices();
      if (indices.getNumElements() != 0) {
        printDenseIntOrFPElementsAttr(indices, /*allowHex=*/false);
        os << ", ";
        printDenseElementsAttr(sparseEltAttr.getValues(), /*allowHex=*/true);
      }
      os << '>';
    }

  } else if (auto locAttr = attr.dyn_cast<LocationAttr>()) {
    printLocation(locAttr);

  } else {
    return printDialectAttribute(attr);
  }

  // Don't print the type if we must elide it, or if it is a None type.
  if (typeElision != AttrTypeElision::Must && !attrType.isa<NoneType>()) {
    os << " : ";
    printType(attrType);
  }
}

/// Print the integer element of a DenseElementsAttr.
static void printDenseIntElement(const APInt &value, raw_ostream &os,
                                 bool isSigned) {
  if (value.getBitWidth() == 1)
    os << (value.getBoolValue() ? "true" : "false");
  else
    value.print(os, isSigned);
}

static void
printDenseElementsAttrImpl(bool isSplat, ShapedType type, raw_ostream &os,
                           function_ref<void(unsigned)> printEltFn) {
  // Special case for 0-d and splat tensors.
  if (isSplat)
    return printEltFn(0);

  // Special case for degenerate tensors.
  auto numElements = type.getNumElements();
  if (numElements == 0)
    return;

  // We use a mixed-radix counter to iterate through the shape. When we bump a
  // non-least-significant digit, we emit a close bracket. When we next emit an
  // element we re-open all closed brackets.

  // The mixed-radix counter, with radices in 'shape'.
  int64_t rank = type.getRank();
  SmallVector<unsigned, 4> counter(rank, 0);
  // The number of brackets that have been opened and not closed.
  unsigned openBrackets = 0;

  auto shape = type.getShape();
  auto bumpCounter = [&] {
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
    printEltFn(idx);
    bumpCounter();
  }
  while (openBrackets-- > 0)
    os << ']';
}

void ModulePrinter::printDenseElementsAttr(DenseElementsAttr attr,
                                           bool allowHex) {
  if (auto stringAttr = attr.dyn_cast<DenseStringElementsAttr>())
    return printDenseStringElementsAttr(stringAttr);

  printDenseIntOrFPElementsAttr(attr.cast<DenseIntOrFPElementsAttr>(),
                                allowHex);
}

void ModulePrinter::printDenseIntOrFPElementsAttr(DenseIntOrFPElementsAttr attr,
                                                  bool allowHex) {
  auto type = attr.getType();
  auto elementType = type.getElementType();

  // Check to see if we should format this attribute as a hex string.
  auto numElements = type.getNumElements();
  if (!attr.isSplat() && allowHex &&
      shouldPrintElementsAttrWithHex(numElements)) {
    ArrayRef<char> rawData = attr.getRawData();
    if (llvm::support::endian::system_endianness() ==
        llvm::support::endianness::big) {
      // Convert endianess in big-endian(BE) machines. `rawData` is BE in BE
      // machines. It is converted here to print in LE format.
      SmallVector<char, 64> outDataVec(rawData.size());
      MutableArrayRef<char> convRawData(outDataVec);
      DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
          rawData, convRawData, type);
      os << '"' << "0x"
         << llvm::toHex(StringRef(convRawData.data(), convRawData.size()))
         << "\"";
    } else {
      os << '"' << "0x"
         << llvm::toHex(StringRef(rawData.data(), rawData.size())) << "\"";
    }

    return;
  }

  if (ComplexType complexTy = elementType.dyn_cast<ComplexType>()) {
    Type complexElementType = complexTy.getElementType();
    // Note: The if and else below had a common lambda function which invoked
    // printDenseElementsAttrImpl. This lambda was hitting a bug in gcc 9.1,9.2
    // and hence was replaced.
    if (complexElementType.isa<IntegerType>()) {
      bool isSigned = !complexElementType.isUnsignedInteger();
      printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
        auto complexValue = *(attr.getComplexIntValues().begin() + index);
        os << "(";
        printDenseIntElement(complexValue.real(), os, isSigned);
        os << ",";
        printDenseIntElement(complexValue.imag(), os, isSigned);
        os << ")";
      });
    } else {
      printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
        auto complexValue = *(attr.getComplexFloatValues().begin() + index);
        os << "(";
        printFloatValue(complexValue.real(), os);
        os << ",";
        printFloatValue(complexValue.imag(), os);
        os << ")";
      });
    }
  } else if (elementType.isIntOrIndex()) {
    bool isSigned = !elementType.isUnsignedInteger();
    auto intValues = attr.getIntValues();
    printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
      printDenseIntElement(*(intValues.begin() + index), os, isSigned);
    });
  } else {
    assert(elementType.isa<FloatType>() && "unexpected element type");
    auto floatValues = attr.getFloatValues();
    printDenseElementsAttrImpl(attr.isSplat(), type, os, [&](unsigned index) {
      printFloatValue(*(floatValues.begin() + index), os);
    });
  }
}

void ModulePrinter::printDenseStringElementsAttr(DenseStringElementsAttr attr) {
  ArrayRef<StringRef> data = attr.getRawStringData();
  auto printFn = [&](unsigned index) {
    os << "\"";
    printEscapedString(data[index], os);
    os << "\"";
  };
  printDenseElementsAttrImpl(attr.isSplat(), attr.getType(), os, printFn);
}

void ModulePrinter::printType(Type type) {
  if (!type) {
    os << "<<NULL TYPE>>";
    return;
  }

  // Try to print an alias for this type.
  if (state && succeeded(state->getAliasState().getAlias(type, os)))
    return;

  TypeSwitch<Type>(type)
      .Case<OpaqueType>([&](OpaqueType opaqueTy) {
        printDialectSymbol(os, "!", opaqueTy.getDialectNamespace(),
                           opaqueTy.getTypeData());
      })
      .Case<IndexType>([&](Type) { os << "index"; })
      .Case<BFloat16Type>([&](Type) { os << "bf16"; })
      .Case<Float16Type>([&](Type) { os << "f16"; })
      .Case<Float32Type>([&](Type) { os << "f32"; })
      .Case<Float64Type>([&](Type) { os << "f64"; })
      .Case<IntegerType>([&](IntegerType integerTy) {
        if (integerTy.isSigned())
          os << 's';
        else if (integerTy.isUnsigned())
          os << 'u';
        os << 'i' << integerTy.getWidth();
      })
      .Case<FunctionType>([&](FunctionType funcTy) {
        os << '(';
        interleaveComma(funcTy.getInputs(), [&](Type ty) { printType(ty); });
        os << ") -> ";
        ArrayRef<Type> results = funcTy.getResults();
        if (results.size() == 1 && !results[0].isa<FunctionType>()) {
          os << results[0];
        } else {
          os << '(';
          interleaveComma(results, [&](Type ty) { printType(ty); });
          os << ')';
        }
      })
      .Case<VectorType>([&](VectorType vectorTy) {
        os << "vector<";
        for (int64_t dim : vectorTy.getShape())
          os << dim << 'x';
        os << vectorTy.getElementType() << '>';
      })
      .Case<RankedTensorType>([&](RankedTensorType tensorTy) {
        os << "tensor<";
        for (int64_t dim : tensorTy.getShape()) {
          if (ShapedType::isDynamic(dim))
            os << '?';
          else
            os << dim;
          os << 'x';
        }
        os << tensorTy.getElementType() << '>';
      })
      .Case<UnrankedTensorType>([&](UnrankedTensorType tensorTy) {
        os << "tensor<*x";
        printType(tensorTy.getElementType());
        os << '>';
      })
      .Case<MemRefType>([&](MemRefType memrefTy) {
        os << "memref<";
        for (int64_t dim : memrefTy.getShape()) {
          if (ShapedType::isDynamic(dim))
            os << '?';
          else
            os << dim;
          os << 'x';
        }
        printType(memrefTy.getElementType());
        for (auto map : memrefTy.getAffineMaps()) {
          os << ", ";
          printAttribute(AffineMapAttr::get(map));
        }
        // Only print the memory space if it is the non-default one.
        if (memrefTy.getMemorySpace())
          os << ", " << memrefTy.getMemorySpace();
        os << '>';
      })
      .Case<UnrankedMemRefType>([&](UnrankedMemRefType memrefTy) {
        os << "memref<*x";
        printType(memrefTy.getElementType());
        // Only print the memory space if it is the non-default one.
        if (memrefTy.getMemorySpace())
          os << ", " << memrefTy.getMemorySpace();
        os << '>';
      })
      .Case<ComplexType>([&](ComplexType complexTy) {
        os << "complex<";
        printType(complexTy.getElementType());
        os << '>';
      })
      .Case<TupleType>([&](TupleType tupleTy) {
        os << "tuple<";
        interleaveComma(tupleTy.getTypes(),
                        [&](Type type) { printType(type); });
        os << '>';
      })
      .Case<NoneType>([&](Type) { os << "none"; })
      .Default([&](Type type) { return printDialectType(type); });
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
  void printValueID(Value value, bool printResultNo = true,
                    raw_ostream *streamOverride = nullptr) const;

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
  void printOperand(Value value, raw_ostream &os) override {
    printValueID(value, /*printResultNo=*/true, &os);
  }

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

  /// Print the given affine map with the symbol and dimension operands printed
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
  // Output the aliases at the top level that can't be deferred.
  state->getAliasState().printNonDeferredAliases(os, newLine);

  // Print the module.
  print(op.getOperation());
  os << newLine;

  // Output the aliases at the top level that can be deferred.
  state->getAliasState().printDeferredAliases(os, newLine);
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
      os << "  // block is not in a region!";
    } else if (block->hasNoPredecessors()) {
      os << "  // no predecessors";
    } else if (auto *pred = block->getSinglePredecessor()) {
      os << "  // pred: ";
      printBlockName(pred);
    } else {
      // We want to print the predecessors in increasing numeric order, not in
      // whatever order the use-list is in, so gather and sort them.
      SmallVector<std::pair<unsigned, Block *>, 4> predIDs;
      for (auto *pred : block->getPredecessors())
        predIDs.push_back({state->getSSANameState().getBlockID(pred), pred});
      llvm::array_pod_sort(predIDs.begin(), predIDs.end());

      os << "  // " << predIDs.size() << " preds: ";

      interleaveComma(predIDs, [&](std::pair<unsigned, Block *> pred) {
        printBlockName(pred.second);
      });
    }
    os << newLine;
  }

  currentIndent += indentWidth;
  auto range = llvm::make_range(
      block->begin(), std::prev(block->end(), printBlockTerminator ? 0 : 1));
  for (auto &op : range) {
    print(&op);
    os << newLine;
  }
  currentIndent -= indentWidth;
}

void OperationPrinter::printValueID(Value value, bool printResultNo,
                                    raw_ostream *streamOverride) const {
  state->getSSANameState().printValueID(value, printResultNo,
                                        streamOverride ? *streamOverride : os);
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
  if (!expr) {
    os << "<<NULL AFFINE EXPR>>";
    return;
  }
  ModulePrinter(os).printAffineExpr(*this);
}

void AffineExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void AffineMap::print(raw_ostream &os) const {
  if (!map) {
    os << "<<NULL AFFINE MAP>>";
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
  BlockArgument arg = this->cast<BlockArgument>();
  os << "<block argument> of type '" << arg.getType()
     << "' at index: " << arg.getArgNumber() << '\n';
}
void Value::print(raw_ostream &os, AsmState &state) {
  if (auto *op = getDefiningOp())
    return op->print(os, state);

  // TODO: Improve this.
  BlockArgument arg = this->cast<BlockArgument>();
  os << "<block argument> of type '" << arg.getType()
     << "' at index: " << arg.getArgNumber() << '\n';
}

void Value::dump() {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void Value::printAsOperand(raw_ostream &os, AsmState &state) {
  // TODO: This doesn't necessarily capture all potential cases.
  // Currently, region arguments can be shadowed when printing the main
  // operation. If the IR hasn't been printed, this will produce the old SSA
  // name and not the shadowed name.
  state.getImpl().getSSANameState().printValueID(*this, /*printResultNo=*/true,
                                                 os);
}

void Operation::print(raw_ostream &os, OpPrintingFlags flags) {
  // Find the operation to number from based upon the provided flags.
  Operation *printedOp = this;
  bool shouldUseLocalScope = flags.shouldUseLocalScope();
  do {
    // If we are printing local scope, stop at the first operation that is
    // isolated from above.
    if (shouldUseLocalScope && printedOp->isKnownIsolatedFromAbove())
      break;

    // Otherwise, traverse up to the next parent.
    Operation *parentOp = printedOp->getParentOp();
    if (!parentOp)
      break;
    printedOp = parentOp;
  } while (true);

  AsmState state(printedOp);
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
    state.getImpl().initializeAliases(*this, flags);
  print(os, state, flags);
}
void ModuleOp::print(raw_ostream &os, AsmState &state, OpPrintingFlags flags) {
  OperationPrinter(os, flags, state.getImpl()).print(*this);
}

void ModuleOp::dump() { print(llvm::errs()); }
