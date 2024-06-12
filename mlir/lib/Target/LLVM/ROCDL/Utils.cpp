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
static std::optional<llvm::msgpack::Document>
getAMDHSANote(llvm::object::ELFObjectFile<ELFT> &elfObj) {
  using namespace llvm;
  using namespace llvm::object;
  using namespace llvm::ELF;
  const ELFFile<ELFT> &elf = elfObj.getELFFile();
  auto secOrErr = elf.sections();
  if (!secOrErr)
    return std::nullopt;
  ArrayRef<typename ELFT::Shdr> sections = *secOrErr;
  for (auto section : sections) {
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
      msgpack::Document msgPackDoc;
      if (!msgPackDoc.readFromBlob(msgPackString, /*Multi=*/false))
        return std::nullopt;
      if (msgPackDoc.getRoot().isScalar())
        return std::nullopt;
      return std::optional<llvm::msgpack::Document>(std::move(msgPackDoc));
    }
  }
  return std::nullopt;
}

/// Return the `amdhsa.kernels` metadata in the ELF object or std::nullopt on
/// failure. This is a helper function that casts a generic `ObjectFile` to the
/// appropiate `ELFObjectFile`.
static std::optional<llvm::msgpack::Document>
getAMDHSANote(ArrayRef<char> elfData) {
  using namespace llvm;
  using namespace llvm::object;
  if (elfData.empty())
    return std::nullopt;
  MemoryBufferRef buffer(StringRef(elfData.data(), elfData.size()), "buffer");
  Expected<std::unique_ptr<ObjectFile>> objOrErr =
      ObjectFile::createELFObjectFile(buffer);
  if (!objOrErr || !objOrErr.get()) {
    // Drop the error.
    llvm::consumeError(objOrErr.takeError());
    return std::nullopt;
  }
  ObjectFile &elf = *(objOrErr.get());
  std::optional<llvm::msgpack::Document> metadata;
  if (auto *obj = dyn_cast<ELF32LEObjectFile>(&elf))
    metadata = getAMDHSANote(*obj);
  else if (auto *obj = dyn_cast<ELF32BEObjectFile>(&elf))
    metadata = getAMDHSANote(*obj);
  else if (auto *obj = dyn_cast<ELF64LEObjectFile>(&elf))
    metadata = getAMDHSANote(*obj);
  else if (auto *obj = dyn_cast<ELF64BEObjectFile>(&elf))
    metadata = getAMDHSANote(*obj);
  return metadata;
}

/// Utility functions for converting `llvm::msgpack::DocNode` nodes.
static Attribute convertNode(Builder &builder, llvm::msgpack::DocNode &node);
static Attribute convertNode(Builder &builder,
                             llvm::msgpack::MapDocNode &node) {
  NamedAttrList attrs;
  for (auto kv : node) {
    if (!kv.first.isString())
      continue;
    if (Attribute attr = convertNode(builder, kv.second)) {
      auto key = kv.first.getString();
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
      auto kind = n.getKind();
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
static llvm::StringMap<DictionaryAttr> getELFMetadata(Builder &builder,
                                                      ArrayRef<char> elfData) {
  std::optional<llvm::msgpack::Document> metadata = getAMDHSANote(elfData);
  if (!metadata)
    return {};
  llvm::StringMap<DictionaryAttr> kernelMD;
  llvm::msgpack::DocNode &root = (metadata)->getRoot();
  // Fail if `root` is not a map -it should be for AMD Obj Ver 3.
  if (!root.isMap())
    return kernelMD;
  auto &kernels = root.getMap()["amdhsa.kernels"];
  // Fail if `amdhsa.kernels` is not an array.
  if (!kernels.isArray())
    return kernelMD;
  // Convert each of the kernels.
  for (auto &kernel : kernels.getArray()) {
    if (!kernel.isMap())
      continue;
    auto &kernelMap = kernel.getMap();
    auto &name = kernelMap[".name"];
    if (!name.isString())
      continue;
    NamedAttrList attrList;
    // Convert the kernel properties.
    for (auto kv : kernelMap) {
      if (!kv.first.isString())
        continue;
      StringRef key = kv.first.getString();
      key.consume_front(".");
      key.consume_back(".");
      if (key == "name")
        continue;
      if (Attribute attr = convertNode(builder, kv.second))
        attrList.append(key, attr);
    }
    if (!attrList.empty())
      kernelMD[name.getString()] = builder.getDictionaryAttr(attrList);
  }
  return kernelMD;
}

gpu::KernelTableAttr
mlir::ROCDL::getAMDHSAKernelsMetadata(Operation *gpuModule,
                                      ArrayRef<char> elfData) {
  auto module = cast<gpu::GPUModuleOp>(gpuModule);
  Builder builder(module.getContext());
  NamedAttrList moduleAttrs;
  llvm::StringMap<DictionaryAttr> mdMap = getELFMetadata(builder, elfData);
  for (auto funcOp : module.getBody()->getOps<LLVM::LLVMFuncOp>()) {
    if (!funcOp->getDiscardableAttr("rocdl.kernel"))
      continue;
    moduleAttrs.append(
        funcOp.getName(),
        gpu::KernelAttr::get(funcOp, mdMap.lookup(funcOp.getName())));
  }
  return gpu::KernelTableAttr::get(
      moduleAttrs.getDictionary(module.getContext()));
}
