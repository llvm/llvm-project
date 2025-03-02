//===-- Properties.td - Properties definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base properties defination file.
//
//===----------------------------------------------------------------------===//

#ifndef PROPERTIES
#define PROPERTIES

include "mlir/IR/Constraints.td"
include "mlir/IR/Utils.td"

// Base class for defining properties.
class Property<string storageTypeParam = "", string desc = ""> {
  // User-readable one line summary used in error reporting messages. If empty,
  // a generic message will be used.
  string summary = desc;
  // The full description of this property.
  string description = "";
  code storageType = storageTypeParam;
  code interfaceType = storageTypeParam;

  // The expression to convert from the storage type to the Interface
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format:
  // - `$_storage` will contain the property in the storage type.
  code convertFromStorage = "$_storage";

  // The call expression to build a property storage from the interface type.
  //
  // Format:
  // - `$_storage` will contain the property in the storage type.
  // - `$_value` will contain the property in the user interface type.
  code assignToStorage = "$_storage = $_value";

  // The call expression to convert from the storage type to an attribute.
  // The resulting attribute must be non-null in non-error cases.
  //
  // Format:
  // - `$_storage` is the storage type value.
  // - `$_ctxt` is a `MLIRContext *`.
  //
  // The expression must return an `Attribute` and will be used as a function body.
  code convertToAttribute = [{
    return convertToAttribute($_ctxt, $_storage);
  }];

  // The call expression to convert from an Attribute to the storage type.
  //
  // Format:
  // - `$_storage` is a reference to a value of the storage type.
  // - `$_attr` is the attribute.
  // - `$_diag` is a callback to get a Diagnostic to emit error.
  //
  // The expression must return a LogicalResult and will be used as a function body
  // or in other similar contexts.
  code convertFromAttribute = [{
    return convertFromAttribute($_storage, $_attr, $_diag);
  }];

  // The verification predicate for this property. Defaults to the true predicate,
  // since properties are always their expected type.
  // Within the predicate, `$_self` is an instance of the **interface**
  // type of the property. Setting this field to ? will also result in a
  // true predicate but is not recommended, as it breaks composability.
  Pred predicate = TruePred;

  // The call expression to hash the property.
  //
  // Format:
  // - `$_storage` is the variable to hash.
  //
  // The expression should define a llvm::hash_code.
  // If unspecified, defaults to `llvm::hash_value($_storage)`.
  // The default is not specified in tablegen because many combinators, like
  // ArrayProperty, can fall back to more efficient implementations of
  // `hashProperty` when their underlying elements have trivial hashing.
  code hashProperty = "";

  // The body of the parser for a value of this property.
  // Format:
  // - `$_parser` is the OpAsmParser.
  // - `$_storage` is the location into which the value is to be placed if it is
  //  present.
  // - `$_ctxt` is a `MLIRContext *`
  //
  // This defines the body of a function (typically a lambda) that returns a
  // ParseResult. There is an implicit `return success()` at the end of the parser
  // code.
  //
  // When this code executes, `$_storage` will be initialized to the property's
  // default value (if any, accounting for the storage type override).
  code parser = [{
    auto value = ::mlir::FieldParser<}] # storageType # [{>::parse($_parser);
    if (::mlir::failed(value))
      return ::mlir::failure();
    $_storage = std::move(*value);
  }];

  // The body of the parser for a value of this property as the anchor of an optional
  // group. This should parse the property if possible and do nothing if a value of
  // the relevant type is not next in the parse stream.
  // You are not required to define this parser if it cannot be meaningfully
  // implemented.
  // This has the same context and substitutions as `parser` except that it is
  // required to return an OptionalParseResult.
  //
  // If the optional parser doesn't parse anything, it should not set
  // $_storage, since the parser doesn't know if the default value has been
  // overwritten.
  code optionalParser = "";

  // The printer for a value of this property.
  // Format:
  // - `$_storage` is the storage data.
  // - `$_printer` is the OpAsmPrinter instance.
  // - `$_ctxt` is a `MLIRContext *`
  //
  // This may be called in an expression context, so variable declarations must
  // be placed within a new scope.
  //
  // The printer for a property should always print a non-empty value - default value
  // printing elision happens outside the context of this printing expression.
  code printer = "$_printer << $_storage";

  // The call expression to emit the storage type to bytecode.
  //
  // Format:
  // - `$_storage` is the storage type value.
  // - `$_writer` is a `DialectBytecodeWriter`.
  // - `$_ctxt` is a `MLIRContext *`.
  //
  // This will become the body af a function returning void.
  code writeToMlirBytecode = [{
    writeToMlirBytecode($_writer, $_storage);
  }];

  // The call expression to read the storage type from bytecode.
  //
  // Format:
  // - `$_storage` is the storage type value.
  // - `$_reader` is a `DialectBytecodeReader`.
  // - `$_ctxt` is a `MLIRContext *`.
  //
  // This will become the body of a function returning LogicalResult.
  // There is an implicit `return success()` at the end of this function.
  //
  // When this code executes, `$_storage` will be initialized to the property's
  // default value (if any, accounting for the storage type override).
  code readFromMlirBytecode = [{
    if (::mlir::failed(readFromMlirBytecode($_reader, $_storage)))
      return ::mlir::failure();
  }];

  // Base definition for the property. Used to look through `OptionalProperty`
  // for some format generation, as with the `baseAttr` field on attributes.
  Property baseProperty = ?;

  // Default value for the property within its storage. This should be an expression
  // of type `interfaceType` and should be comparable with other types of that
  // interface typ with `==`. The empty string means there is no default value.
  string defaultValue = "";

  // If set, the default value the storage of the property should be initilized to.
  // This is only needed when the storage and interface types of the property
  // are distinct (ex. SmallVector for storage vs. ArrayRef for interfacing), as it
  // will fall back to `defaultValue` when unspecified.
  string storageTypeValueOverride = "";
}

/// Implementation of the Property class's `readFromMlirBytecode` field using
/// the default `convertFromAttribute` implementation.
/// Users not wanting to implement their own `readFromMlirBytecode` and
/// `writeToMlirBytecode` implementations can opt into using this implementation
/// by writing:
///
/// let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;
/// let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
///
/// in their property definition.
/// Serialization and deserialization is performed using the attributes
/// returned by `convertFromAttribute` and `convertToAttribute`.
///
/// WARNING: This implementation creates a less than optimal encoding.
/// Users caring about optimal encoding should not use this implementation and
/// implement `readFromMlirBytecode` and `writeToMlirBytecode` themselves.
defvar readMlirBytecodeUsingConvertFromAttribute = [{
  ::mlir::Attribute attr;
  if (::mlir::failed($_reader.readAttribute(attr)))
    return ::mlir::failure();
  if (::mlir::failed(convertFromAttribute($_storage, attr, nullptr)))
    return ::mlir::failure();
}];

/// Implementation of the Property class's `writeToMlirBytecode` field using
/// the default `convertToAttribute` implementation.
/// See description of `readMlirBytecodeUsingConvertFromAttribute` above for
/// details.
defvar writeMlirBytecodeWithConvertToAttribute = [{
  $_writer.writeAttribute(convertToAttribute($_ctxt, $_storage))
}];

//===----------------------------------------------------------------------===//
// Primitive property kinds

// Any kind of integer stored as properties.
class IntProp<string storageTypeParam, string desc = ""> :
    Property<storageTypeParam, desc> {
  let summary = !if(!empty(desc), storageTypeParam, desc);
  let optionalParser = [{
    return $_parser.parseOptionalInteger($_storage);
  }];
  let writeToMlirBytecode = [{
    $_writer.writeVarInt($_storage);
  }];
  let readFromMlirBytecode = [{
    uint64_t val;
    if (failed($_reader.readVarInt(val)))
      return ::mlir::failure();
    $_storage = val;
  }];
}

class IntProperty<string storageTypeParam, string desc = "">
  : IntProp<storageTypeParam, desc>, Deprecated<"moved to the shorter name IntProp">;

def I32Prop : IntProp<"int32_t">;
def I64Prop : IntProp<"int64_t">;

def I32Property : IntProp<"int32_t">, Deprecated<"moved to shorter name I32Prop">;
def I64Property : IntProp<"int64_t">, Deprecated<"moved to shorter name I64Prop">;

class EnumProp<string storageTypeParam, string desc = "", string default = ""> :
    Property<storageTypeParam, desc> {
  // TODO:  implement predicate for enum validity.
  let writeToMlirBytecode = [{
    $_writer.writeVarInt(static_cast<uint64_t>($_storage));
  }];
  let readFromMlirBytecode = [{
    uint64_t val;
    if (failed($_reader.readVarInt(val)))
      return ::mlir::failure();
    $_storage = static_cast<}] # storageTypeParam # [{>(val);
  }];
  let defaultValue = default;
}

class EnumProperty<string storageTypeParam, string desc = "", string default = ""> :
  EnumProp<storageTypeParam, desc, default>,
  Deprecated<"moved to shorter name EnumProp">;

// Note: only a class so we can deprecate the old name
class _cls_StringProp : Property<"std::string", "string"> {
  let interfaceType = "::llvm::StringRef";
  let convertFromStorage = "::llvm::StringRef{$_storage}";
  let assignToStorage = "$_storage = $_value.str()";
  let optionalParser = [{
    if (::mlir::failed($_parser.parseOptionalString(&$_storage)))
      return std::nullopt;
  }];
  let printer = "$_printer.printString($_storage)";
  let readFromMlirBytecode = [{
    StringRef val;
    if (::mlir::failed($_reader.readString(val)))
      return ::mlir::failure();
    $_storage = val.str();
  }];
  let writeToMlirBytecode = [{
    $_writer.writeOwnedString($_storage);
  }];
}
def StringProp : _cls_StringProp;
def StringProperty : _cls_StringProp, Deprecated<"moved to shorter name StringProp">;

// Note: only a class so we can deprecate the old name
class _cls_BoolProp : IntProp<"bool", "boolean"> {
  let printer = [{ $_printer << ($_storage ? "true" : "false") }];
  let readFromMlirBytecode = [{
    return $_reader.readBool($_storage);
  }];
  let writeToMlirBytecode = [{
    $_writer.writeOwnedBool($_storage);
  }];
}
def BoolProp : _cls_BoolProp;
def BoolProperty : _cls_BoolProp, Deprecated<"moved to shorter name BoolProp">;

// Note: only a class so we can deprecate the old name
class _cls_UnitProp : Property<"bool", "unit property"> {
  let summary = "unit property";
  let description = [{
    A property whose presence or abscence is used as a flag.

    This is stored as a boolean that defaults to false, and is named UnitProperty
    by analogy with UnitAttr, which has the more comprehensive rationale and
    explains the less typical syntax.

    Note that this attribute does have a syntax for the false case to allow for its
    use in contexts where default values shouldn't be elided.
  }];
  let defaultValue = "false";

  let convertToAttribute = [{
    if ($_storage)
      return ::mlir::UnitAttr::get($_ctxt);
    else
      return ::mlir::BoolAttr::get($_ctxt, false);
  }];
  let convertFromAttribute = [{
    if (::llvm::isa<::mlir::UnitAttr>($_attr)) {
      $_storage = true;
      return ::mlir::success();
    }
    if (auto boolAttr = ::llvm::dyn_cast<::mlir::BoolAttr>($_attr)) {
      $_storage = boolAttr.getValue();
      return ::mlir::success();
    }
    return ::mlir::failure();
  }];

  let parser = [{
    ::llvm::StringRef keyword;
    if (::mlir::failed($_parser.parseOptionalKeyword(&keyword,
        {"unit", "unit_absent"})))
      return $_parser.emitError($_parser.getCurrentLocation(),
        "expected 'unit' or 'unit_absent'");
    $_storage = (keyword == "unit");
  }];

  let optionalParser = [{
    ::llvm::StringRef keyword;
    if (::mlir::failed($_parser.parseOptionalKeyword(&keyword,
        {"unit", "unit_absent"})))
      return std::nullopt;
    $_storage = (keyword == "unit");
  }];

  let printer = [{
    $_printer << ($_storage ? "unit" : "unit_absent")
  }];

  let writeToMlirBytecode = [{
    $_writer.writeOwnedBool($_storage);
  }];
  let readFromMlirBytecode = [{
    if (::mlir::failed($_reader.readBool($_storage)))
      return ::mlir::failure();
  }];
}
def UnitProp : _cls_UnitProp;
def UnitProperty : _cls_UnitProp, Deprecated<"moved to shorter name UnitProp">;

//===----------------------------------------------------------------------===//
// Property field overwrites

/// Class for giving a property a default value.
/// This doesn't change anything about the property other than giving it a default
/// which can be used by ODS to elide printing.
class DefaultValuedProp<Property p, string default = "", string storageDefault = ""> : Property<p.storageType, p.summary> {
  let defaultValue = default;
  let storageTypeValueOverride = storageDefault;
  let baseProperty = p;
  // Keep up to date with `Property` above.
  let summary = p.summary;
  let description = p.description;
  let storageType = p.storageType;
  let interfaceType = p.interfaceType;
  let convertFromStorage = p.convertFromStorage;
  let assignToStorage = p.assignToStorage;
  let convertToAttribute = p.convertToAttribute;
  let convertFromAttribute = p.convertFromAttribute;
  let predicate = p.predicate;
  let hashProperty = p.hashProperty;
  let parser = p.parser;
  let optionalParser = p.optionalParser;
  let printer = p.printer;
  let readFromMlirBytecode = p.readFromMlirBytecode;
  let writeToMlirBytecode = p.writeToMlirBytecode;
}
class DefaultValuedProperty<Property p, string default = "", string storageDefault = "">
  : DefaultValuedProp<p, default, storageDefault>, Deprecated<"moved to shorter name DefaultValuedProp">;

/// Apply the predicate `pred` to the property `p`, ANDing it with any
/// predicates it may already have. If `newSummary` is provided, replace the
/// summary of `p` with `newSummary`.
class ConfinedProp<Property p, Pred pred, string newSummary = "">
    : Property<p.storageType, !if(!empty(newSummary), p.summary, newSummary)> {
  let predicate = !if(!ne(p.predicate, TruePred), And<[p.predicate, pred]>, pred);
  let baseProperty = p;
  // Keep up to date with `Property` above.
  let description = p.description;
  let storageType = p.storageType;
  let interfaceType = p.interfaceType;
  let convertFromStorage = p.convertFromStorage;
  let assignToStorage = p.assignToStorage;
  let convertToAttribute = p.convertToAttribute;
  let convertFromAttribute = p.convertFromAttribute;
  let hashProperty = p.hashProperty;
  let parser = p.parser;
  let optionalParser = p.optionalParser;
  let printer = p.printer;
  let readFromMlirBytecode = p.readFromMlirBytecode;
  let writeToMlirBytecode = p.writeToMlirBytecode;
  let defaultValue = p.defaultValue;
  let storageTypeValueOverride = p.storageTypeValueOverride;
}

class ConfinedProperty<Property p, Pred pred, string newSummary = "">
  : ConfinedProp<p, pred, newSummary>,
    Deprecated<"moved to shorter name ConfinedProp">;

//===----------------------------------------------------------------------===//
// Primitive property combinators

/// Create a variable named `name` of `prop`'s storage type that is initialized
/// to the correct default value, if there is one.
class _makePropStorage<Property prop, string name> {
  code ret = prop.storageType # " " # name
      # !cond(!not(!empty(prop.storageTypeValueOverride)) : " = " # prop.storageTypeValueOverride,
        !not(!empty(prop.defaultValue)) : " = " # prop.defaultValue,
        true : "") # ";";
}

/// Construct a `Pred`icate `ret` that wraps the predicate of the underlying
/// property `childProp` with:
///
///   [](childProp.storageType& s) {
///     return [](childProp.interfaceType i) {
///       return leafSubst(childProp.predicate, "$_self" to "i");
///     }(childProp.convertFromStorage(s))
///   }
///
/// and then appends `prefix` and `suffix`.
class _makeStorageWrapperPred<Property wrappedProp> {
  Pred ret =
    Concat<
      "[](" # "const " # wrappedProp.storageType
        # "& baseStore) -> bool { return []("
        # wrappedProp.interfaceType # " baseIface) -> bool { return (",
      SubstLeaves<"$_self", "baseIface", wrappedProp.predicate>,
      "); }(" # !subst("$_storage", "baseStore", wrappedProp.convertFromStorage)
        # "); }">;
}

/// The generic class for arrays of some other property, which is stored as a
/// `SmallVector` of that property. This uses an `ArrayAttr` as its attribute form
/// though subclasses can override this, as is the case with IntArrayAttr below.
/// Those wishing to use a non-default number of SmallVector elements should
/// subclass `ArrayProp`.
class ArrayProp<Property elem = Property<>, string newSummary = ""> :
  Property<"::llvm::SmallVector<" # elem.storageType # ">",
    !if(!empty(newSummary), "array of " # elem.summary, newSummary)> {
  let interfaceType = "::llvm::ArrayRef<" # elem.storageType # ">";
  let convertFromStorage = "::llvm::ArrayRef<" # elem.storageType # ">{$_storage}";
  let assignToStorage = "$_storage.assign($_value.begin(), $_value.end())";

  let convertFromAttribute = [{
    auto arrayAttr = ::llvm::dyn_cast_if_present<::mlir::ArrayAttr>($_attr);
    if (!arrayAttr)
      return $_diag() << "expected array attribute";
    for (::mlir::Attribute elemAttr : arrayAttr) {
      }] # _makePropStorage<elem, "elemVal">.ret # [{
      auto elemRes = [&](Attribute propAttr, }] # elem.storageType # [{& propStorage) -> ::mlir::LogicalResult {
        }] # !subst("$_attr", "propAttr",
          !subst("$_storage", "propStorage", elem.convertFromAttribute)) # [{
      }(elemAttr, elemVal);
      if (::mlir::failed(elemRes))
        return ::mlir::failure();
      $_storage.push_back(std::move(elemVal));
    }
    return ::mlir::success();
  }];

  let convertToAttribute = [{
    SmallVector<Attribute> elems;
    for (const auto& elemVal : $_storage) {
      auto elemAttr = [&](const }] # elem.storageType #[{& propStorage) -> ::mlir::Attribute {
        }] # !subst("$_storage", "propStorage", elem.convertToAttribute) # [{
      }(elemVal);
      elems.push_back(elemAttr);
    }
    return ::mlir::ArrayAttr::get($_ctxt, elems);
  }];

  let predicate = !if(!eq(elem.predicate, TruePred),
    TruePred,
    Concat<"::llvm::all_of($_self, ", _makeStorageWrapperPred<elem>.ret, ")">);

  defvar theParserBegin = [{
    auto& storage = $_storage;
    auto parseElemFn = [&]() -> ::mlir::ParseResult {
      }] # _makePropStorage<elem, "elemVal">.ret # [{
      auto elemParse = [&](}] # elem.storageType # [{& propStorage) -> ::mlir::ParseResult {
        }] # !subst("$_storage", "propStorage", elem.parser) # [{
        return ::mlir::success();
       }(elemVal);
      if (::mlir::failed(elemParse))
        return ::mlir::failure();
      storage.push_back(std::move(elemVal));
      return ::mlir::success();
    };
    }];
  let parser = theParserBegin # [{
    return $_parser.parseCommaSeparatedList(
      ::mlir::OpAsmParser::Delimiter::Square, parseElemFn);
  }];
  // Hack around the lack of a peek method
  let optionalParser = theParserBegin # [{
    auto oldLoc = $_parser.getCurrentLocation();
    auto parseResult = $_parser.parseCommaSeparatedList(
      ::mlir::OpAsmParser::Delimiter::OptionalSquare, parseElemFn);
    if (::mlir::failed(parseResult))
      return ::mlir::failure();
    auto newLoc = $_parser.getCurrentLocation();
    if (oldLoc == newLoc)
      return std::nullopt;
    return ::mlir::success();
  }];

  let printer = [{ [&](){
    $_printer << "[";
    auto elemPrinter = [&](const }] # elem.storageType # [{& elemVal) {
      }] # !subst("$_storage", "elemVal", elem.printer) #[{;
    };
    ::llvm::interleaveComma($_storage, $_printer, elemPrinter);
    $_printer << "]";
  }()}];

  let readFromMlirBytecode = [{
    uint64_t length;
    if (::mlir::failed($_reader.readVarInt(length)))
      return ::mlir::failure();
    $_storage.reserve(length);
    for (uint64_t i = 0; i < length; ++i) {
      }]# _makePropStorage<elem, "elemVal">.ret # [{
      auto elemRead = [&](}] # elem.storageType # [{& propStorage) -> ::mlir::LogicalResult {
        }] # !subst("$_storage", "propStorage", elem.readFromMlirBytecode) # [{;
        return ::mlir::success();
      }(elemVal);
      if (::mlir::failed(elemRead))
        return ::mlir::failure();
      $_storage.push_back(std::move(elemVal));
    }
  }];

  let writeToMlirBytecode = [{
    $_writer.writeVarInt($_storage.size());
    for (const auto& elemVal : $_storage) {
      [&]() {
        }] # !subst("$_storage", "elemVal", elem.writeToMlirBytecode) #[{;
      }();
    }
  }];

  // There's no hash_value for SmallVector<T>, so we construct the ArrayRef ourselves.
  // In the non-trivial case, we define a mapped range to get internal hash
  // codes.
  let hashProperty = !if(!empty(elem.hashProperty),
    [{::llvm::hash_value(::llvm::ArrayRef<}] # elem.storageType # [{>{$_storage})}],
    [{[&]() -> ::llvm::hash_code {
        auto getElemHash = [](const auto& propStorage) -> ::llvm::hash_code {
          return }] # !subst("$_storage", "propStorage", elem.hashProperty) # [{;
        };
        auto mapped = ::llvm::map_range($_storage, getElemHash);
        return ::llvm::hash_combine_range(mapped.begin(), mapped.end());
      }()
    }]);
}
class ArrayProperty<Property elem = Property<>, string newSummary = "">
  : ArrayProp<elem, newSummary>, Deprecated<"moved to shorter name ArrayProp">;

class IntArrayProp<Property elem, string newSummary=""> :
    ArrayProp<elem, newSummary> {
  // Bring back the trivial conversions we don't get in the general case.
  let convertFromAttribute = [{
    return convertFromAttribute($_storage, $_attr, $_diag);
  }];
  let convertToAttribute = [{
    return convertToAttribute($_ctxt, $_storage);
  }];
}
class IntArrayProperty<Property elem, string newSummary="">
  : IntArrayProp<elem, newSummary>, Deprecated<"moved to shorter name IntArrayProp">;

/// An optional property, stored as an std::optional<p.storageType>
/// interfaced with as an std::optional<p.interfaceType>..
/// The syntax is `none` (or empty string if elided) for an absent value or
/// `some<[underlying property]>` when a value is set.
///
/// As a special exception, if the underlying property has an optional parser and
/// no default value (ex. an integer property), the printer will skip the `some`
/// bracketing and delegate to the optional parser. In that case, the syntax is the
/// syntax of the underlying property, or the keyword `none` in the rare cases that
/// it is needed. This behavior can be disabled by setting `canDelegateParsing` to 0.
class OptionalProp<Property p, bit canDelegateParsing = 1>
    : Property<"std::optional<" # p.storageType # ">", "optional " # p.summary> {

  // In the cases where the underlying attribute is plain old data that's passed by
  // value, the conversion code is trivial.
  defvar hasTrivialStorage = !and(!eq(p.convertFromStorage, "$_storage"),
    !eq(p.assignToStorage, "$_storage = $_value"),
    !eq(p.storageType, p.interfaceType));

  defvar delegatesParsing = !and(!empty(p.defaultValue),
    !not(!empty(p.optionalParser)), canDelegateParsing);

  let interfaceType = "std::optional<" # p.interfaceType # ">";
  let defaultValue = "std::nullopt";

  let convertFromStorage = !if(hasTrivialStorage,
    p.convertFromStorage,
    [{($_storage.has_value() ? std::optional<}] # p.interfaceType # ">{"
      # !subst("$_storage", "(*($_storage))", p.convertFromStorage)
      # [{} : std::nullopt)}]);
  let assignToStorage = !if(hasTrivialStorage,
    p.assignToStorage,
    [{[&]() {
      if (!$_value.has_value()) {
        $_storage = std::nullopt;
        return;
      }
      }] # _makePropStorage<p, "presentVal">.ret # [{
      [&](}] # p.storageType # [{& propStorage) {
        }] # !subst("$_storage", "propStorage",
          !subst("$_value", "(*($_value))", p.assignToStorage)) # [{;
      }(presentVal);
      $_storage = std::move(presentVal);
    }()}]);

  let convertFromAttribute = [{
    auto arrayAttr = ::llvm::dyn_cast<::mlir::ArrayAttr>($_attr);
    if (!arrayAttr)
      return $_diag() << "expected optional properties to materialize as arrays";
    if (arrayAttr.size() > 1)
      return $_diag() << "expected optional properties to become 0- or 1-element arrays";
    if (arrayAttr.empty()) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    ::mlir::Attribute presentAttr = arrayAttr[0];
    }] # _makePropStorage<p, "presentVal">.ret # [{
    auto presentRes = [&](Attribute propAttr, }] # p.storageType # [{& propStorage) -> ::mlir::LogicalResult {
      }] # !subst("$_storage", "propStorage",
          !subst("$_attr", "propAttr", p.convertFromAttribute)) # [{
    }(presentAttr, presentVal);
    if (::mlir::failed(presentRes))
      return ::mlir::failure();
    $_storage = std::move(presentVal);
    return ::mlir::success();
  }];

  let convertToAttribute = [{
    if (!$_storage.has_value()) {
      return ::mlir::ArrayAttr::get($_ctxt, {});
    }
    auto attr = [&]() -> ::mlir::Attribute {
      }] # !subst("$_storage", "(*($_storage))", p.convertToAttribute) # [{
    }();
    return ::mlir::ArrayAttr::get($_ctxt, {attr});
  }];

  let predicate = !if(!ne(p.predicate, TruePred),
    Or<[CPred<"!$_self.has_value()">,
        SubstLeaves<"$_self", "(*($_self))", p.predicate>]>,
    TruePred);

  defvar delegatedParserBegin = [{
    if (::mlir::succeeded($_parser.parseOptionalKeyword("none"))) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    }] #_makePropStorage<p, "presentVal">.ret # [{
    auto delegParseResult = [&](}] # p.storageType # [{& propStorage) -> ::mlir::OptionalParseResult {
    }] # !subst("$_storage", "propStorage", p.optionalParser) # [{
        return ::mlir::success();
    }(presentVal);
    if (!delegParseResult.has_value()) {
  }];

  defvar delegatedParserEnd = [{
    }
    if (delegParseResult.has_value() && ::mlir::failed(*delegParseResult))
      return ::mlir::failure();
    $_storage = std::move(presentVal);
    return ::mlir::success();
  }];
  // If we're being explicitly called for our parser, we're expecting to have been
  // printede into a context where the default value isn't elided. Therefore,
  // not-present from the underlying parser is a failure.
  defvar delegatedParser = delegatedParserBegin # [{
    return ::mlir::failure();
  }] # delegatedParserEnd;
  defvar delegatedOptionalParser = delegatedParserBegin # [{
      return std::nullopt;
  }] # delegatedParserEnd;

  defvar generalParserBegin = [{
    ::llvm::StringRef keyword;
    if (::mlir::failed($_parser.parseOptionalKeyword(&keyword, {"none", "some"}))) {
  }];
  defvar generalParserEnd = [{
    }
    if (keyword == "none") {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    if (::mlir::failed($_parser.parseLess()))
      return ::mlir::failure();
    }] # _makePropStorage<p, "presentVal">.ret # [{
    auto presentParse = [&](}] # p.storageType # [{& propStorage) -> ::mlir::ParseResult {
      }] # !subst("$_storage", "propStorage", p.parser) # [{
      return ::mlir::success();
    }(presentVal);
    if (presentParse || $_parser.parseGreater())
      return ::mlir::failure();
    $_storage = std::move(presentVal);
  }];
  defvar generalParser = generalParserBegin # [{
    return $_parser.emitError($_parser.getCurrentLocation(), "expected 'none' or 'some<prop>'");
  }] # generalParserEnd;
  defvar generalOptionalParser = generalParserBegin # [{
    return std::nullopt;
  }] # generalParserEnd;

  let parser = !if(delegatesParsing, delegatedParser, generalParser);
  let optionalParser = !if(delegatesParsing,
    delegatedOptionalParser, generalOptionalParser);

  defvar delegatedPrinter = [{
    [&]() {
      if (!$_storage.has_value()) {
        $_printer << "none";
        return;
      }
      }] # !subst("$_storage", "(*($_storage))", p.printer) # [{;
    }()}];
  defvar generalPrinter = [{
      [&]() {
        if (!$_storage.has_value()) {
          $_printer << "none";
          return;
        }
        $_printer << "some<";
        }] # !subst("$_storage", "(*($_storage))", p.printer) # [{;
        $_printer << ">";
      }()}];
  let printer = !if(delegatesParsing, delegatedPrinter, generalPrinter);

  let readFromMlirBytecode = [{
    bool isPresent = false;
    if (::mlir::failed($_reader.readBool(isPresent)))
      return ::mlir::failure();
    if (!isPresent) {
      $_storage = std::nullopt;
      return ::mlir::success();
    }
    }] # _makePropStorage<p, "presentVal">.ret # [{
    auto presentResult = [&](}] # p.storageType # [{& propStorage) -> ::mlir::LogicalResult {
      }] # !subst("$_storage", "propStorage", p.readFromMlirBytecode) # [{;
      return ::mlir::success();
    }(presentVal);
    if (::mlir::failed(presentResult))
      return ::mlir::failure();
    $_storage = std::move(presentVal);
  }];
  let writeToMlirBytecode = [{
    $_writer.writeOwnedBool($_storage.has_value());
    if (!$_storage.has_value())
      return;
  }] # !subst("$_storage", "(*($_storage))", p.writeToMlirBytecode);

  let hashProperty = !if(!empty(p.hashProperty), p.hashProperty,
    [{ ::llvm::hash_value($_storage.has_value() ? std::optional<::llvm::hash_code>{}] #
      !subst("$_storage", "(*($_storage))", p.hashProperty) #[{} : std::nullopt) }]);
  assert !or(!not(delegatesParsing), !eq(defaultValue, "std::nullopt")),
    "For delegated parsing to be used, the default value must be nullopt. " #
    "To use a non-trivial default, set the canDelegateParsing argument to 0";
}
class OptionalProperty<Property p, bit canDelegateParsing = 1>
  : OptionalProp<p, canDelegateParsing>,
    Deprecated<"moved to shorter name OptionalProp">;
#endif // PROPERTIES
