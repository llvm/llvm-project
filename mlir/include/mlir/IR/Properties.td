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
  // - `$_ctxt` will contain an `MLIRContext *`.
  code convertFromStorage = "$_storage";

  // The call expression to build a property storage from the interface type.
  //
  // Format:
  // - `$_storage` will contain the property in the storage type.
  // - `$_value` will contain the property in the user interface type.
  code assignToStorage = "$_storage = $_value";

  // The call expression to convert from the storage type to an attribute.
  //
  // Format:
  // - `$_storage` is the storage type value.
  // - `$_ctxt` is a `MLIRContext *`.
  //
  // The expression must result in an Attribute.
  code convertToAttribute = [{
    convertToAttribute($_ctxt, $_storage)
  }];

  // The call expression to convert from an Attribute to the storage type.
  //
  // Format:
  // - `$_storage` is the storage type value.
  // - `$_attr` is the attribute.
  // - `$_diag` is a callback to get a Diagnostic to emit error.
  //
  // The expression must return a LogicalResult
  code convertFromAttribute = [{
    return convertFromAttribute($_storage, $_attr, $_diag);
  }];

  // The call expression to hash the property.
  //
  // Format:
  // - `$_storage` is the variable to hash.
  //
  // The expression should define a llvm::hash_code.
  code hashProperty = [{
    llvm::hash_value($_storage);
  }];

  // The call expression to emit the storage type to bytecode.
  //
  // Format:
  // - `$_storage` is the storage type value.
  // - `$_writer` is a `DialectBytecodeWriter`.
  // - `$_ctxt` is a `MLIRContext *`.
  code writeToMlirBytecode = [{
    writeToMlirBytecode($_writer, $_storage)
  }];

  // The call expression to read the storage type from bytecode.
  //
  // Format:
  // - `$_storage` is the storage type value.
  // - `$_reader` is a `DialectBytecodeReader`.
  // - `$_ctxt` is a `MLIRContext *`.
  code readFromMlirBytecode = [{
    if (::mlir::failed(readFromMlirBytecode($_reader, $_storage)))
      return ::mlir::failure();
  }];

  // Default value for the property.
  string defaultValue = ?;
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
class IntProperty<string storageTypeParam = "", string desc = ""> :
    Property<storageTypeParam, desc> {
  code writeToMlirBytecode = [{
    $_writer.writeVarInt($_storage);
  }];
  code readFromMlirBytecode = [{
    uint64_t val;
    if (failed($_reader.readVarInt(val)))
      return ::mlir::failure();
    $_storage = val;
  }];
}

class ArrayProperty<string storageTypeParam = "", int n, string desc = ""> :
  Property<storageTypeParam # "[" # n # "]", desc> {
  let interfaceType = "::llvm::ArrayRef<" # storageTypeParam # ">";
  let convertFromStorage = "$_storage";
  let assignToStorage = "::llvm::copy($_value, $_storage)";
}

#endif // PROPERTIES
