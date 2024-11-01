//===- DebugImporter.cpp - LLVM to MLIR Debug conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugImporter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

void DebugImporter::translate(llvm::Function *func, LLVMFuncOp funcOp) {
  if (!func->getSubprogram())
    return;

  // Add a fused location to link the subprogram information.
  StringAttr name = StringAttr::get(context, func->getSubprogram()->getName());
  funcOp->setLoc(FusedLocWith<DISubprogramAttr>::get(
      {NameLoc::get(name)}, translate(func->getSubprogram()), context));
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

DIBasicTypeAttr DebugImporter::translateImpl(llvm::DIBasicType *node) {
  return DIBasicTypeAttr::get(context, node->getTag(), node->getName(),
                              node->getSizeInBits(), node->getEncoding());
}

DICompileUnitAttr DebugImporter::translateImpl(llvm::DICompileUnit *node) {
  Optional<DIEmissionKind> emissionKind =
      symbolizeDIEmissionKind(node->getEmissionKind());
  return DICompileUnitAttr::get(context, node->getSourceLanguage(),
                                translate(node->getFile()),
                                StringAttr::get(context, node->getProducer()),
                                node->isOptimized(), emissionKind.value());
}

DICompositeTypeAttr DebugImporter::translateImpl(llvm::DICompositeType *node) {
  Optional<DIFlags> flags = symbolizeDIFlags(node->getFlags());
  SmallVector<DINodeAttr> elements;
  for (llvm::DINode *element : node->getElements()) {
    assert(element && "expected a non-null element type");
    elements.push_back(translate(element));
  }
  return DICompositeTypeAttr::get(
      context, node->getTag(), StringAttr::get(context, node->getName()),
      translate(node->getFile()), node->getLine(), translate(node->getScope()),
      translate(node->getBaseType()), flags.value_or(DIFlags::Zero),
      node->getSizeInBits(), node->getAlignInBits(), elements);
}

DIDerivedTypeAttr DebugImporter::translateImpl(llvm::DIDerivedType *node) {
  return DIDerivedTypeAttr::get(
      context, node->getTag(),
      node->getRawName() ? StringAttr::get(context, node->getName()) : nullptr,
      translate(node->getBaseType()), node->getSizeInBits(),
      node->getAlignInBits(), node->getOffsetInBits());
}

DIFileAttr DebugImporter::translateImpl(llvm::DIFile *node) {
  return DIFileAttr::get(context, node->getFilename(), node->getDirectory());
}

DILexicalBlockAttr DebugImporter::translateImpl(llvm::DILexicalBlock *node) {
  return DILexicalBlockAttr::get(context, translate(node->getScope()),
                                 translate(node->getFile()), node->getLine(),
                                 node->getColumn());
}

DILexicalBlockFileAttr
DebugImporter::translateImpl(llvm::DILexicalBlockFile *node) {
  return DILexicalBlockFileAttr::get(context, translate(node->getScope()),
                                     translate(node->getFile()),
                                     node->getDiscriminator());
}

DILocalVariableAttr DebugImporter::translateImpl(llvm::DILocalVariable *node) {
  return DILocalVariableAttr::get(context, translate(node->getScope()),
                                  StringAttr::get(context, node->getName()),
                                  translate(node->getFile()), node->getLine(),
                                  node->getArg(), node->getAlignInBits(),
                                  translate(node->getType()));
}

DIScopeAttr DebugImporter::translateImpl(llvm::DIScope *node) {
  return cast<DIScopeAttr>(translate(static_cast<llvm::DINode *>(node)));
}

DISubprogramAttr DebugImporter::translateImpl(llvm::DISubprogram *node) {
  Optional<DISubprogramFlags> subprogramFlags =
      symbolizeDISubprogramFlags(node->getSubprogram()->getSPFlags());
  return DISubprogramAttr::get(
      context, translate(node->getUnit()), translate(node->getScope()),
      StringAttr::get(context, node->getName()),
      node->getRawLinkageName()
          ? StringAttr::get(context, node->getLinkageName())
          : nullptr,
      translate(node->getFile()), node->getLine(), node->getScopeLine(),
      subprogramFlags.value(), translate(node->getType()));
}

DISubrangeAttr DebugImporter::translateImpl(llvm::DISubrange *node) {
  auto getIntegerAttrOrNull = [&](llvm::DISubrange::BoundType data) {
    if (auto *constInt = llvm::dyn_cast_or_null<llvm::ConstantInt *>(data))
      return IntegerAttr::get(IntegerType::get(context, 64),
                              constInt->getSExtValue());
    return IntegerAttr();
  };
  return DISubrangeAttr::get(context, getIntegerAttrOrNull(node->getCount()),
                             getIntegerAttrOrNull(node->getLowerBound()),
                             getIntegerAttrOrNull(node->getUpperBound()),
                             getIntegerAttrOrNull(node->getStride()));
}

DISubroutineTypeAttr
DebugImporter::translateImpl(llvm::DISubroutineType *node) {
  // Separate the result type since it is null for void functions.
  DITypeAttr resultType = translate(*node->getTypeArray().begin());
  SmallVector<DITypeAttr> argumentTypes;
  for (llvm::DIType *type : llvm::drop_begin(node->getTypeArray())) {
    assert(type && "expected a non-null argument type");
    argumentTypes.push_back(translate(type));
  }
  return DISubroutineTypeAttr::get(context, node->getCC(), resultType,
                                   argumentTypes);
}

DITypeAttr DebugImporter::translateImpl(llvm::DIType *node) {
  return cast<DITypeAttr>(translate(static_cast<llvm::DINode *>(node)));
}

DINodeAttr DebugImporter::translate(llvm::DINode *node) {
  if (!node)
    return nullptr;

  // Check for a cached instance.
  if (DINodeAttr attr = nodeToAttr.lookup(node))
    return attr;

  // Convert the debug metadata if possible.
  auto translateNode = [this](llvm::DINode *node) -> DINodeAttr {
    if (auto *casted = dyn_cast<llvm::DIBasicType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DICompileUnit>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DICompositeType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIDerivedType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIFile>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DILexicalBlock>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DILexicalBlockFile>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DILocalVariable>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DISubprogram>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DISubrange>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DISubroutineType>(node))
      return translateImpl(casted);
    return nullptr;
  };
  if (DINodeAttr attr = translateNode(node)) {
    nodeToAttr.insert({node, attr});
    return attr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Locations
//===----------------------------------------------------------------------===//

Location DebugImporter::translateLoc(llvm::DILocation *loc) {
  if (!loc)
    return UnknownLoc::get(context);

  // Get the file location of the instruction.
  Location result = FileLineColLoc::get(context, loc->getFilename(),
                                        loc->getLine(), loc->getColumn());

  // Add call site information, if available.
  if (llvm::DILocation *inlinedAt = loc->getInlinedAt())
    result = CallSiteLoc::get(result, translateLoc(inlinedAt));

  // Add scope information.
  assert(loc->getScope() && "expected non-null scope");
  result = FusedLocWith<DIScopeAttr>::get({result}, translate(loc->getScope()),
                                          context);
  return result;
}
