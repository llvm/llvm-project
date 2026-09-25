#ifndef MLIR_INTERFACES_BYTECODEDIALECTINTERFACE
#define MLIR_INTERFACES_BYTECODEDIALECTINTERFACE

include "mlir/IR/Interfaces.td"

def BytecodeDialectInterface : DialectInterface<"BytecodeDialectInterface"> {
  let description = [{}];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Read an attribute belonging to this dialect from the given reader. This
        method should return null in the case of failure. Optionally, the dialect
        version can be accessed through the reader.
      }],
      "::mlir::Attribute", "readAttribute",
      (ins "::mlir::DialectBytecodeReader &":$reader),
      [{
        reader.emitError() << "dialect " << getDialect()->getNamespace()
                           << " does not support reading attributes from bytecode";
        return ::mlir::Attribute();
      }]
    >,
    InterfaceMethod<[{
        Read a type belonging to this dialect from the given reader. This method
        should return null in the case of failure. Optionally, the dialect version
        can be accessed thorugh the reader.
      }],
      "::mlir::Type", "readType", (ins "::mlir::DialectBytecodeReader &":$reader),
      [{
        reader.emitError() << "dialect " << getDialect()->getNamespace()
                           << " does not support reading types from bytecode";
        return ::mlir::Type();
      }]
    >,
    InterfaceMethod<[{
        Write the given attribute, which belongs to this dialect, to the given
        writer. This method may return failure to indicate that the given
        attribute could not be encoded, in which case the textual format will be
        used to encode this attribute instead.
      }],
      "::llvm::LogicalResult", "writeAttribute",
      (ins "::mlir::Attribute":$attribute, "::mlir::DialectBytecodeWriter &":$writer),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Write the given type, which belongs to this dialect, to the given writer.
        This method may return failure to indicate that the given type could not
        be encoded, in which case the textual format will be used to encode this
        type instead.
      }],
      "::llvm::LogicalResult", "writeType",
      (ins "::mlir::Type":$type, "::mlir::DialectBytecodeWriter &":$writer),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Write the version of this dialect to the given writer.
      }],
      "void", "writeVersion",
      (ins "::mlir::DialectBytecodeWriter &":$writer)
    >,
    InterfaceMethod<[{
        Read the version of this dialect from the provided reader and return it as
        a `unique_ptr` to a dialect version object.
      }],
      "std::unique_ptr<::mlir::DialectVersion>", "readVersion",
      (ins "::mlir::DialectBytecodeReader &":$reader),
      [{
        reader.emitError("Dialect does not support versioning");
        return nullptr;
      }]
    >,
    InterfaceMethod<[{
        Hook invoked after parsing completed, if a version directive was present
        and included an entry for the current dialect. This hook offers the
        opportunity to the dialect to visit the IR and upgrades constructs emitted
        by the version of the dialect corresponding to the provided version.
      }],
      "::llvm::LogicalResult", "upgradeFromVersion",
      (ins "::mlir::Operation *":$topLevelOp, "const ::mlir::DialectVersion &":$version),
      [{
        return ::llvm::success();
      }]
    >
  ];
}


#endif
