//===- ModuleSubstreamVisitor.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleSubstreamVisitor.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"

using namespace llvm;
using namespace llvm::codeview;

Error IModuleSubstreamVisitor::visitSymbols(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::Symbols, Data);
}
Error IModuleSubstreamVisitor::visitLines(BinaryStreamRef Data,
                                          const LineSubstreamHeader *Header,
                                          const LineInfoArray &Lines) {
  return visitUnknown(ModuleSubstreamKind::Lines, Data);
}
Error IModuleSubstreamVisitor::visitStringTable(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::StringTable, Data);
}
Error IModuleSubstreamVisitor::visitFileChecksums(
    BinaryStreamRef Data, const FileChecksumArray &Checksums) {
  return visitUnknown(ModuleSubstreamKind::FileChecksums, Data);
}
Error IModuleSubstreamVisitor::visitFrameData(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::FrameData, Data);
}
Error IModuleSubstreamVisitor::visitInlineeLines(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::InlineeLines, Data);
}
Error IModuleSubstreamVisitor::visitCrossScopeImports(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::CrossScopeExports, Data);
}
Error IModuleSubstreamVisitor::visitCrossScopeExports(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::CrossScopeImports, Data);
}
Error IModuleSubstreamVisitor::visitILLines(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::ILLines, Data);
}
Error IModuleSubstreamVisitor::visitFuncMDTokenMap(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::FuncMDTokenMap, Data);
}
Error IModuleSubstreamVisitor::visitTypeMDTokenMap(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::TypeMDTokenMap, Data);
}
Error IModuleSubstreamVisitor::visitMergedAssemblyInput(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::MergedAssemblyInput, Data);
}
Error IModuleSubstreamVisitor::visitCoffSymbolRVA(BinaryStreamRef Data) {
  return visitUnknown(ModuleSubstreamKind::CoffSymbolRVA, Data);
}

Error llvm::codeview::visitModuleSubstream(const ModuleSubstream &R,
                                           IModuleSubstreamVisitor &V) {
  switch (R.getSubstreamKind()) {
  case ModuleSubstreamKind::Symbols:
    return V.visitSymbols(R.getRecordData());
  case ModuleSubstreamKind::Lines: {
    BinaryStreamReader Reader(R.getRecordData());
    const LineSubstreamHeader *Header;
    if (auto EC = Reader.readObject(Header))
      return EC;
    VarStreamArrayExtractor<LineColumnEntry> E(Header);
    LineInfoArray LineInfos(E);
    if (auto EC = Reader.readArray(LineInfos, Reader.bytesRemaining()))
      return EC;
    return V.visitLines(R.getRecordData(), Header, LineInfos);
  }
  case ModuleSubstreamKind::StringTable:
    return V.visitStringTable(R.getRecordData());
  case ModuleSubstreamKind::FileChecksums: {
    BinaryStreamReader Reader(R.getRecordData());
    FileChecksumArray Checksums;
    if (auto EC = Reader.readArray(Checksums, Reader.bytesRemaining()))
      return EC;
    return V.visitFileChecksums(R.getRecordData(), Checksums);
  }
  case ModuleSubstreamKind::FrameData:
    return V.visitFrameData(R.getRecordData());
  case ModuleSubstreamKind::InlineeLines:
    return V.visitInlineeLines(R.getRecordData());
  case ModuleSubstreamKind::CrossScopeImports:
    return V.visitCrossScopeImports(R.getRecordData());
  case ModuleSubstreamKind::CrossScopeExports:
    return V.visitCrossScopeExports(R.getRecordData());
  case ModuleSubstreamKind::ILLines:
    return V.visitILLines(R.getRecordData());
  case ModuleSubstreamKind::FuncMDTokenMap:
    return V.visitFuncMDTokenMap(R.getRecordData());
  case ModuleSubstreamKind::TypeMDTokenMap:
    return V.visitTypeMDTokenMap(R.getRecordData());
  case ModuleSubstreamKind::MergedAssemblyInput:
    return V.visitMergedAssemblyInput(R.getRecordData());
  case ModuleSubstreamKind::CoffSymbolRVA:
    return V.visitCoffSymbolRVA(R.getRecordData());
  default:
    return V.visitUnknown(R.getSubstreamKind(), R.getRecordData());
  }
}
