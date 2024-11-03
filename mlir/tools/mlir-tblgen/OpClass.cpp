//===- OpClass.cpp - Implementation of an Op Class ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpClass.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// OpClass definitions
//===----------------------------------------------------------------------===//

OpClass::OpClass(StringRef name, StringRef extraClassDeclaration,
                 std::string extraClassDefinition)
    : Class(name.str()), extraClassDeclaration(extraClassDeclaration),
      extraClassDefinition(std::move(extraClassDefinition)),
      parent(addParent("::mlir::Op")) {
  parent.addTemplateParam(getClassName().str());
  declare<VisibilityDeclaration>(Visibility::Public);
  /// Inherit functions from Op.
  declare<UsingDeclaration>("Op::Op");
  declare<UsingDeclaration>("Op::print");
  /// Type alias for the adaptor class.
  declare<UsingDeclaration>("Adaptor", className + "Adaptor");
  declare<UsingDeclaration>("GenericAdaptor",
                            className + "GenericAdaptor<RangeT>")
      ->addTemplateParam("RangeT");
  declare<UsingDeclaration>(
      "FoldAdaptor", "GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>");
}

void OpClass::finalize() {
  Class::finalize();
  declare<VisibilityDeclaration>(Visibility::Public);
  declare<ExtraClassDeclaration>(extraClassDeclaration, extraClassDefinition);
}
