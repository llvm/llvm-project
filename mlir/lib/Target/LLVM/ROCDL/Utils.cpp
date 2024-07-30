//===- Utils.cpp - MLIR ROCDL target utils ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines ROCDL target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/AMDGPUMetadata.h"

using namespace mlir;
using namespace mlir::ROCDL;

/// Search the ELF object and return an object containing the `amdhsa.kernels`
/// metadata note. Function adapted from:
/// llvm-project/llvm/tools/llvm-readobj/ELFDumper.cpp Also see
/// `amdhsa.kernels`:
/// https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-metadata
template <typename ELFT>
static std::unique_ptr<llvm::msgpack::Document>
getAMDHSANote(llvm::object::ELFObjectFile<ELFT> &elfObj) {
  using namespace llvm;
  using namespace llvm::object;
  using namespace llvm::ELF;
  const ELFFile<ELFT> &elf = elfObj.getELFFile();
  Expected<typename ELFT::ShdrRange> secOrErr = elf.sections();
  if (!secOrErr) {
    consumeError(secOrErr.takeError());
    return nullptr;
  }
  ArrayRef<typename ELFT::Shdr> sections = *secOrErr;
  for (const typename ELFT::Shdr &section : sections) {
    if (section.sh_type != ELF::SHT_NOTE)
      continue;
    size_t align = std::max(static_cast<unsigned>(section.sh_addralign), 4u);
    Error err = Error::success();
    for (const typename ELFT::Note note : elf.notes(section, err)) {
      StringRef name = note.getName();
      if (name != "AMDGPU")
        continue;
      uint32_t type = note.getType();
      if (type != ELF::NT_AMDGPU_METADATA)
        continue;
      ArrayRef<uint8_t> desc = note.getDesc(align);
      StringRef msgPackString =
          StringRef(reinterpret_cast<const char *>(desc.data()), desc.size());
      std::unique_ptr<llvm::msgpack::Document> msgPackDoc(
          new llvm::msgpack::Document());
      if (!msgPackDoc->readFromBlob(msgPackString, /*Multi=*/false))
        return nullptr;
      if (msgPackDoc->getRoot().isScalar())
        return nullptr;
      return msgPackDoc;
    }
  }
  return nullptr;
}

/// Return the `amdhsa.kernels` metadata in the ELF object or nullptr on
/// failure. This is a helper function that casts a generic `ObjectFile` to the
/// appropiate `ELFObjectFile`.
static std::unique_ptr<llvm::msgpack::Document>
getAMDHSANote(ArrayRef<char> elfData) {
  using namespace llvm;
  using namespace llvm::object;
  if (elfData.empty())
    return nullptr;
  MemoryBufferRef buffer(StringRef(elfData.data(), elfData.size()), "buffer");
  Expected<std::unique_ptr<ObjectFile>> objOrErr =
      ObjectFile::createELFObjectFile(buffer);
  if (!objOrErr || !objOrErr.get()) {
    // Drop the error.
    llvm::consumeError(objOrErr.takeError());
    return nullptr;
  }
  ObjectFile &elf = *(objOrErr.get());
  if (auto *obj = dyn_cast<ELF32LEObjectFile>(&elf))
    return getAMDHSANote(*obj);
  else if (auto *obj = dyn_cast<ELF32BEObjectFile>(&elf))
    return getAMDHSANote(*obj);
  else if (auto *obj = dyn_cast<ELF64LEObjectFile>(&elf))
    return getAMDHSANote(*obj);
  else if (auto *obj = dyn_cast<ELF64BEObjectFile>(&elf))
    return getAMDHSANote(*obj);
  return nullptr;
}

/// Utility functions for converting `llvm::msgpack::DocNode` nodes.
static Attribute convertNode(Builder &builder, llvm::msgpack::DocNode &node);
static Attribute convertNode(Builder &builder,
                             llvm::msgpack::MapDocNode &node) {
  NamedAttrList attrs;
  for (auto &[keyNode, valueNode] : node) {
    if (!keyNode.isString())
      continue;
    StringRef key = keyNode.getString();
    if (Attribute attr = convertNode(builder, valueNode)) {
      key.consume_front(".");
      key.consume_back(".");
      attrs.append(key, attr);
    }
  }
  if (attrs.empty())
    return nullptr;
  return builder.getDictionaryAttr(attrs);
}

static Attribute convertNode(Builder &builder,
                             llvm::msgpack::ArrayDocNode &node) {
  using NodeKind = llvm::msgpack::Type;
  // Use `DenseIntAttr` if we know all the attrs are ints.
  if (llvm::all_of(node, [](llvm::msgpack::DocNode &n) {
        auto kind = n.getKind();
        return kind == NodeKind::Int || kind == NodeKind::UInt;
      })) {
    SmallVector<int64_t> values;
    for (llvm::msgpack::DocNode &n : node) {
      llvm::msgpack::Type kind = n.getKind();
      if (kind == NodeKind::Int)
        values.push_back(n.getInt());
      else if (kind == NodeKind::UInt)
        values.push_back(n.getUInt());
    }
    return builder.getDenseI64ArrayAttr(values);
  }
  // Convert the array.
  SmallVector<Attribute> attrs;
  for (llvm::msgpack::DocNode &n : node) {
    if (Attribute attr = convertNode(builder, n))
      attrs.push_back(attr);
  }
  if (attrs.empty())
    return nullptr;
  return builder.getArrayAttr(attrs);
}

static Attribute convertNode(Builder &builder, llvm::msgpack::DocNode &node) {
  using namespace llvm::msgpack;
  using NodeKind = llvm::msgpack::Type;
  switch (node.getKind()) {
  case NodeKind::Int:
    return builder.getI64IntegerAttr(node.getInt());
  case NodeKind::UInt:
    return builder.getI64IntegerAttr(node.getUInt());
  case NodeKind::Boolean:
    return builder.getI64IntegerAttr(node.getBool());
  case NodeKind::String:
    return builder.getStringAttr(node.getString());
  case NodeKind::Array:
    return convertNode(builder, node.getArray());
  case NodeKind::Map:
    return convertNode(builder, node.getMap());
  default:
    return nullptr;
  }
}

/// The following function should succeed for Code object V3 and above.
std::optional<DenseMap<StringAttr, NamedAttrList>>
mlir::ROCDL::getAMDHSAKernelsELFMetadata(Builder &builder,
                                         ArrayRef<char> elfData) {
  using namespace llvm::msgpack;
  std::unique_ptr<llvm::msgpack::Document> metadata = getAMDHSANote(elfData);
  if (!metadata)
    return std::nullopt;
  DenseMap<StringAttr, NamedAttrList> kernelMD;
  DocNode &rootNode = (metadata)->getRoot();
  // Fail if `rootNode` is not a map -it should be for AMD Obj Ver 3.
  if (!rootNode.isMap())
    return std::nullopt;
  DocNode &kernels = rootNode.getMap()["amdhsa.kernels"];
  // Fail if `amdhsa.kernels` is not an array.
  if (!kernels.isArray())
    return std::nullopt;
  // Convert each of the kernels.
  for (DocNode &kernel : kernels.getArray()) {
    if (!kernel.isMap())
      continue;
    MapDocNode &kernelMap = kernel.getMap();
    DocNode &nameNode = kernelMap[".name"];
    if (!nameNode.isString())
      continue;
    StringRef name = nameNode.getString();
    NamedAttrList attrList;
    // Convert the kernel properties.
    for (auto &[keyNode, valueNode] : kernelMap) {
      if (!keyNode.isString())
        continue;
      StringRef key = keyNode.getString();
      key.consume_front(".");
      key.consume_back(".");
      if (key == "name")
        continue;
      if (Attribute attr = convertNode(builder, valueNode))
        attrList.append(key, attr);
    }
    if (!attrList.empty())
      kernelMD[builder.getStringAttr(name)] = std::move(attrList);
  }
  return kernelMD;
}

gpu::KernelTableAttr mlir::ROCDL::getKernelMetadata(Operation *gpuModule,
                                                    ArrayRef<char> elfData) {
  auto module = cast<gpu::GPUModuleOp>(gpuModule);
  Builder builder(module.getContext());
  SmallVector<gpu::KernelAttr> kernels;
  std::optional<DenseMap<StringAttr, NamedAttrList>> mdMapOrNull =
      getAMDHSAKernelsELFMetadata(builder, elfData);
  for (auto funcOp : module.getBody()->getOps<LLVM::LLVMFuncOp>()) {
    if (!funcOp->getDiscardableAttr("rocdl.kernel"))
      continue;
    kernels.push_back(gpu::KernelAttr::get(
        funcOp, mdMapOrNull ? builder.getDictionaryAttr(
                                  mdMapOrNull->lookup(funcOp.getNameAttr()))
                            : nullptr));
  }
  return gpu::KernelTableAttr::get(gpuModule->getContext(), kernels);
}
