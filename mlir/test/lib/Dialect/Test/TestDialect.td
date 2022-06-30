//===-- TestDialect.td - Test dialect definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_DIALECT
#define TEST_DIALECT

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
  let hasOperationInterfaceFallback = 1;
  let hasNonDefaultDestructor = 1;
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  let isExtensible = 1;
  let dependentDialects = ["::mlir::DLTIDialect"];

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();

    // Provides a custom printing/parsing for some operations.
    ::llvm::Optional<ParseOpHook>
      getParseOperationHook(::llvm::StringRef opName) const override;
    ::llvm::unique_function<void(::mlir::Operation *,
                                 ::mlir::OpAsmPrinter &printer)>
     getOperationPrinter(::mlir::Operation *op) const override;

     ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
     void printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const override;

    /// Returns the external elements data manager for this dialect.
    TestExternalElementsDataManager &getExternalDataManager() {
      return externalDataManager;
    }

  private:
    // Storage for a custom fallback interface.
    void *fallbackEffectOpInterfaces;

    ::mlir::Type parseTestType(::mlir::AsmParser &parser,
                               ::llvm::SetVector<::mlir::Type> &stack) const;
    void printTestType(::mlir::Type type, ::mlir::AsmPrinter &printer,
                       ::llvm::SetVector<::mlir::Type> &stack) const;

    /// An external data manager used to test external elements data.
    TestExternalElementsDataManager externalDataManager;
  }];
}

#endif // TEST_DIALECT
