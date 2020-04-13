//===-- LayoutUtils.cpp - Decorate composite type with layout information -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Utilities used to get alignment and layout information
// for types in SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"

using namespace mlir;

spirv::StructType
VulkanLayoutUtils::decorateType(spirv::StructType structType) {
  Size size = 0;
  Size alignment = 1;
  return decorateType(structType, size, alignment);
}

spirv::StructType
VulkanLayoutUtils::decorateType(spirv::StructType structType,
                                VulkanLayoutUtils::Size &size,
                                VulkanLayoutUtils::Size &alignment) {
  if (structType.getNumElements() == 0) {
    return structType;
  }

  SmallVector<Type, 4> memberTypes;
  SmallVector<Size, 4> layoutInfo;
  SmallVector<spirv::StructType::MemberDecorationInfo, 4> memberDecorations;

  Size structMemberOffset = 0;
  Size maxMemberAlignment = 1;

  for (uint32_t i = 0, e = structType.getNumElements(); i < e; ++i) {
    Size memberSize = 0;
    Size memberAlignment = 1;

    auto memberType =
        decorateType(structType.getElementType(i), memberSize, memberAlignment);
    structMemberOffset = llvm::alignTo(structMemberOffset, memberAlignment);
    memberTypes.push_back(memberType);
    layoutInfo.push_back(structMemberOffset);
    // If the member's size is the max value, it must be the last member and it
    // must be a runtime array.
    assert(memberSize != std::numeric_limits<Size>().max() ||
           (i + 1 == e &&
            structType.getElementType(i).isa<spirv::RuntimeArrayType>()));
    // According to the Vulkan spec:
    // "A structure has a base alignment equal to the largest base alignment of
    // any of its members."
    structMemberOffset += memberSize;
    maxMemberAlignment = std::max(maxMemberAlignment, memberAlignment);
  }

  // According to the Vulkan spec:
  // "The Offset decoration of a member must not place it between the end of a
  // structure or an array and the next multiple of the alignment of that
  // structure or array."
  size = llvm::alignTo(structMemberOffset, maxMemberAlignment);
  alignment = maxMemberAlignment;
  structType.getMemberDecorations(memberDecorations);
  return spirv::StructType::get(memberTypes, layoutInfo, memberDecorations);
}

Type VulkanLayoutUtils::decorateType(Type type, VulkanLayoutUtils::Size &size,
                                     VulkanLayoutUtils::Size &alignment) {
  if (type.isa<spirv::ScalarType>()) {
    alignment = getScalarTypeAlignment(type);
    // Vulkan spec does not specify any padding for a scalar type.
    size = alignment;
    return type;
  }

  switch (type.getKind()) {
  case spirv::TypeKind::Struct:
    return decorateType(type.cast<spirv::StructType>(), size, alignment);
  case spirv::TypeKind::Array:
    return decorateType(type.cast<spirv::ArrayType>(), size, alignment);
  case StandardTypes::Vector:
    return decorateType(type.cast<VectorType>(), size, alignment);
  case spirv::TypeKind::RuntimeArray:
    size = std::numeric_limits<Size>().max();
    return decorateType(type.cast<spirv::RuntimeArrayType>(), alignment);
  default:
    llvm_unreachable("unhandled SPIR-V type");
  }
}

Type VulkanLayoutUtils::decorateType(VectorType vectorType,
                                     VulkanLayoutUtils::Size &size,
                                     VulkanLayoutUtils::Size &alignment) {
  const auto numElements = vectorType.getNumElements();
  auto elementType = vectorType.getElementType();
  Size elementSize = 0;
  Size elementAlignment = 1;

  auto memberType = decorateType(elementType, elementSize, elementAlignment);
  // According to the Vulkan spec:
  // 1. "A two-component vector has a base alignment equal to twice its scalar
  // alignment."
  // 2. "A three- or four-component vector has a base alignment equal to four
  // times its scalar alignment."
  size = elementSize * numElements;
  alignment = numElements == 2 ? elementAlignment * 2 : elementAlignment * 4;
  return VectorType::get(numElements, memberType);
}

Type VulkanLayoutUtils::decorateType(spirv::ArrayType arrayType,
                                     VulkanLayoutUtils::Size &size,
                                     VulkanLayoutUtils::Size &alignment) {
  const auto numElements = arrayType.getNumElements();
  auto elementType = arrayType.getElementType();
  Size elementSize = 0;
  Size elementAlignment = 1;

  auto memberType = decorateType(elementType, elementSize, elementAlignment);
  // According to the Vulkan spec:
  // "An array has a base alignment equal to the base alignment of its element
  // type."
  size = elementSize * numElements;
  alignment = elementAlignment;
  return spirv::ArrayType::get(memberType, numElements, elementSize);
}

Type VulkanLayoutUtils::decorateType(spirv::RuntimeArrayType arrayType,
                                     VulkanLayoutUtils::Size &alignment) {
  auto elementType = arrayType.getElementType();
  Size elementSize = 0;

  auto memberType = decorateType(elementType, elementSize, alignment);
  return spirv::RuntimeArrayType::get(memberType, elementSize);
}

VulkanLayoutUtils::Size
VulkanLayoutUtils::getScalarTypeAlignment(Type scalarType) {
  // According to the Vulkan spec:
  // 1. "A scalar of size N has a scalar alignment of N."
  // 2. "A scalar has a base alignment equal to its scalar alignment."
  // 3. "A scalar, vector or matrix type has an extended alignment equal to its
  // base alignment."
  auto bitWidth = scalarType.getIntOrFloatBitWidth();
  if (bitWidth == 1)
    return 1;
  return bitWidth / 8;
}

bool VulkanLayoutUtils::isLegalType(Type type) {
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    return true;
  }

  auto storageClass = ptrType.getStorageClass();
  auto structType = ptrType.getPointeeType().dyn_cast<spirv::StructType>();
  if (!structType) {
    return true;
  }

  switch (storageClass) {
  case spirv::StorageClass::Uniform:
  case spirv::StorageClass::StorageBuffer:
  case spirv::StorageClass::PushConstant:
  case spirv::StorageClass::PhysicalStorageBuffer:
    return structType.hasLayout() || !structType.getNumElements();
  default:
    return true;
  }
}
