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

// Op partitions, each generated and compiled separately. Ops from TestOps.td
// are in `Core` unless a `let` block or their base class picks another
// partition. All ops from TestOpsSyntax.td are in `Syntax`, which is generated
// from that file.
def Test_CorePartition : Partition<"Core">;
def Test_AttributesPartition : Partition<"Attributes">;
def Test_TraitsPartition : Partition<"Traits">;
def Test_PatternsPartition : Partition<"Patterns">;
def Test_ConversionPartition : Partition<"Conversion">;
def Test_InterfacesPartition : Partition<"Interfaces">;
def Test_PropertiesPartition : Partition<"Properties">;
def Test_SyntaxPartition : Partition<"Syntax">;

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
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
  let opPartitions = [
    Test_CorePartition, Test_AttributesPartition, Test_TraitsPartition,
    Test_PatternsPartition, Test_ConversionPartition, Test_InterfacesPartition,
    Test_PropertiesPartition, Test_SyntaxPartition
  ];
  let defaultOpPartition = Test_CorePartition;
  let dependentDialects = [
    "::mlir::DLTIDialect",
    "::mlir::bufferization::BufferizationDialect"
  ];
  let discardableAttrs = (ins
     "mlir::IntegerAttr":$discardable_attr_key,
     "SimpleAAttr":$other_discardable_attr_key
  );

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerInterfaces();
    void registerTypes();

    // Provides a custom printing/parsing for some operations.
    ::std::optional<ParseOpHook>
      getParseOperationHook(::llvm::StringRef opName) const override;
    ::llvm::unique_function<void(::mlir::Operation *,
                                 ::mlir::OpAsmPrinter &printer)>
     getOperationPrinter(::mlir::Operation *op) const override;

     ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
     void printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const override;

  private:
    // Storage for a custom fallback interface.
    void *fallbackEffectOpInterfaces;
  }];
}

#endif // TEST_DIALECT
