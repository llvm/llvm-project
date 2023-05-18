//===- OpenMPCommon.cpp - Utils for translating MLIR dialect to LLVM IR----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines general utilities for MLIR Dialect translations to LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/OpenMPCommon.h"

llvm::Constant *
mlir::LLVM::createSourceLocStrFromLocation(Location loc,
                                           llvm::OpenMPIRBuilder &builder,
                                           StringRef name, uint32_t &strLen) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    StringRef fileName = fileLoc.getFilename();
    unsigned lineNo = fileLoc.getLine();
    unsigned colNo = fileLoc.getColumn();
    return builder.getOrCreateSrcLocStr(name, fileName, lineNo, colNo, strLen);
  }
  std::string locStr;
  llvm::raw_string_ostream locOS(locStr);
  locOS << loc;
  return builder.getOrCreateSrcLocStr(locOS.str(), strLen);
}

llvm::Constant *
mlir::LLVM::createMappingInformation(Location loc,
                                     llvm::OpenMPIRBuilder &builder) {
  uint32_t strLen;
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    StringRef name = nameLoc.getName();
    return createSourceLocStrFromLocation(nameLoc.getChildLoc(), builder, name,
                                          strLen);
  }
  return createSourceLocStrFromLocation(loc, builder, "unknown", strLen);
}
