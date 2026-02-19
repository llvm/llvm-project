#ifndef MLIR_INTERFACES_LLVMTRANSLATIONDIALECTINTERFACE
#define MLIR_INTERFACES_LLVMTRANSLATIONDIALECTINTERFACE

include "mlir/IR/Interfaces.td"

def LLVMTranslationDialectInterface : DialectInterface<"LLVMTranslationDialectInterface"> {
  let description = [{
    Base class for dialect interfaces providing translation to LLVM IR.
    Dialects that can be translated should provide an implementation of this
    interface for the supported operations. The interface may be implemented in
    a separate library to avoid the "main" dialect library depending on LLVM IR.
    The interface can be attached using the delayed registration mechanism
    available in DialectRegistry.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Hook for derived dialect interface to provide translation of the
        operations to LLVM IR.
      }],
      "::mlir::LogicalResult", "convertOperation",
      (ins "::mlir::Operation *":$op, "::llvm::IRBuilderBase &":$builder,
           "::mlir::LLVM::ModuleTranslation &":$moduleTranslation),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interface to act on an operation that has dialect
        attributes from the derived dialect (the operation itself may be from a
        different dialect). This gets called after the operation has been
        translated. The hook is expected to use moduleTranslation to look up the
        translation results and amend the corresponding IR constructs. Does
        nothing and succeeds by default.
      }],
      "::mlir::LogicalResult", "amendOperation",
      (ins "::mlir::Operation *":$op,
           "::mlir::ArrayRef<::llvm::Instruction *>":$instructions,
           "::mlir::NamedAttribute":$attribute,
           "::mlir::LLVM::ModuleTranslation &":$moduleTranslation),
      [{
        return ::llvm::success();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interface to translate or act on a derived
        dialect attribute that appears on a function parameter. This gets called
        after the function operation has been translated.
      }],
      "::mlir::LogicalResult", "convertParameterAttr",
      (ins "::mlir::LLVM::LLVMFuncOp":$function, "int":$argIdx,
           "::mlir::NamedAttribute":$attr,
           "::mlir::LLVM::ModuleTranslation &":$moduleTranslation),
      [{
        return ::llvm::success();
      }]
    >
  ];
}


#endif
