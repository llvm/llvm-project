#ifndef MLIR_INTERFACES_LLVMIMPORTDIALECTINTERFACE
#define MLIR_INTERFACES_LLVMIMPORTDIALECTINTERFACE

include "mlir/IR/Interfaces.td"

def LLVMImportDialectInterface : DialectInterface<"LLVMImportDialectInterface"> {
  let description = [{
    Base class for dialect interfaces used to import LLVM IR. Dialects that can
    be imported should provide an implementation of this interface for the
    supported intrinsics. The interface may be implemented in a separate library
    to avoid the "main" dialect library depending on LLVM IR. The interface can
    be attached using the delayed registration mechanism available in
    DialectRegistry.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Hook for derived dialect interfaces to implement the import of
        intrinsics into MLIR.
      }],
      "::llvm::LogicalResult", "convertIntrinsic",
      (ins "::mlir::OpBuilder &":$builder, "::llvm::CallInst *":$inst,
           "::mlir::LLVM::ModuleImport &":$moduleImport),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to implement the import of
        instructions into MLIR.
      }],
      "::llvm::LogicalResult", "convertInstruction",
      (ins "::mlir::OpBuilder &":$builder, "::llvm::Instruction *":$inst,
           "::mlir::ArrayRef<llvm::Value *>":$llvmOperands,
           "::mlir::LLVM::ModuleImport &":$moduleImport),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to implement the import of metadata
        into MLIR. Attaches the converted metadata kind and node to the provided
        operation.
      }],
      "::mlir::LogicalResult", "setMetadataAttrs",
      (ins "::mlir::OpBuilder &":$builder, "unsigned":$kind,
           "::llvm::MDNode *":$node, "::mlir::Operation *":$op,
           "::mlir::LLVM::ModuleImport &":$moduleImport),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to publish the supported intrinsics.
        As every LLVM IR intrinsic has a unique integer identifier, the function
        returns the list of supported intrinsic identifiers.
      }],
      "::mlir::ArrayRef<unsigned>", "getSupportedIntrinsics", (ins),
      [{
        return {};
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to publish the supported instructions.
        As every LLVM IR instruction has a unique integer identifier, the function
        returns the list of supported instruction identifiers. These identifiers
        will then be used to match LLVM instructions to the appropriate import
        interface and `convertInstruction` method. It is an error to have multiple
        interfaces overriding the same instruction.
      }],
      "::mlir::ArrayRef<unsigned>", "getSupportedInstructions", (ins),
      [{
        return {};
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to publish the supported metadata
        kinds. As every metadata kind has a unique integer identifier, the
        function returns the list of supported metadata identifiers. The
        `llvmContext` parameter is used to obtain identifiers for metadata kinds
        that do not have a fixed static identifier. Since different LLVM contexts
        can assign different identifiers to these non-static metadata kinds, the
        function must recompute the list of supported metadata identifiers on each
        call.
      }],
      "::mlir::SmallVector<unsigned>", "getSupportedMetadata",
      (ins "::llvm::LLVMContext &":$llvmContext),
      [{
        return {};
      }]
    >
  ];
}

#endif
