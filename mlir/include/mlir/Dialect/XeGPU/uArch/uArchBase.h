//===- uArch.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Base uArch definition for different architectures, plus the SPIRV / Khronos
// OpenCL extension instruction defaults shared across Intel Xe uArchs.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UARCH_UARCHBASE_H
#define MLIR_DIALECT_XEGPU_UARCH_UARCHBASE_H

#include <cassert>
#include <optional>
#include <tuple>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace xegpu {
namespace uArch {

// An enum class to represent the scope of an instruction
enum class InstructionScope { Lane, Subgroup, Workgroup, Cluster };
enum class InstructionKind {
  SubgroupMatrixMultiplyAcc, // Dot Product Accumulate Systolic (DPAS) is a
                             // matrix multiply-add operation
  SubgroupScaledMatrixMultiplyAcc, // Scaled Matrix Multiply Accumulate is a
                                   // DPAS with scaling factor applied to
                                   // operand A or B before multiplication
  Subgroup2DBlockStore,            // Subgroup-level 2D block write instruction
  Subgroup2DBlockLoad,             // Subgroup-level 2D block load instruction
  Subgroup2DBlockPrefetch, // Subgroup-level 2D block prefetch instruction
  StoreScatter,            // Lane-level store (scalar, vector)
  LoadGather,              // Lane-level load (scalar, vector)
};

// A struct to represent basic information about an instruction.
// The primary purpose of the Instruction struct is to provide a generic way to
// represent information about an instruction and to use this information to
// generate the uArch. Specifc instruction in a uArch can inherit from this
// struct and add more fields as needed.
struct Instruction {
  Instruction(InstructionKind kind, InstructionScope scope)
      : instKind(kind), scope(scope) {}

  ~Instruction() = default;
  // Get methods
  InstructionKind getInstructionKind() const { return instKind; }
  InstructionScope getScope() const { return scope; }
  static llvm::StringRef toString(InstructionKind instKind) {
    switch (instKind) {
    case InstructionKind::SubgroupMatrixMultiplyAcc:
      return "dpas";
    case InstructionKind::SubgroupScaledMatrixMultiplyAcc:
      return "dpas_mx";
    case InstructionKind::Subgroup2DBlockStore:
      return "store_nd";
    case InstructionKind::Subgroup2DBlockLoad:
      return "load_nd";
    case InstructionKind::Subgroup2DBlockPrefetch:
      return "prefetch_nd";
    case InstructionKind::StoreScatter:
      return "store";
    case InstructionKind::LoadGather:
      return "load";
    }
    llvm_unreachable("Unknown InstructionKind");
  }

protected:
  const InstructionKind instKind; // Specific InstructionKind (e.g., DPAS)
  const InstructionScope scope;   // scope of the instruction (e.g., lane,
                                  // subgroup, workgroup, cluster)
};

struct uArch {
  enum class Kind {
    // Xe2 family
    Xe2_First,
    PVC = Xe2_First,
    BMG,
    Xe2_Last = BMG,
    Xe3_First,
    CRI = Xe3_First,
    Xe3_Last = CRI
  };

  // Constructor
  uArch(Kind kind, llvm::ArrayRef<const Instruction *> instructionRegistry)
      : kind(kind) {
    for (const Instruction *instr : instructionRegistry)
      this->instructionRegistry[instr->getInstructionKind()] = instr;
  }
  virtual ~uArch() = default;
  Kind getKind() const { return kind; }

  virtual int getSubgroupSize() const = 0;
  virtual unsigned getGeneralPackedFormatBitSize() const = 0;

  const Instruction *getInstruction(InstructionKind instKind) const {
    auto it = instructionRegistry.find(instKind);
    assert(it != instructionRegistry.end() &&
           "Instruction not found in registry");
    return it->second;
  }

  bool isSupportedInstruction(InstructionKind instr) const {
    return instructionRegistry.contains(instr);
  }

protected:
  Kind kind;
  llvm::SmallDenseMap<InstructionKind, const Instruction *, 32>
      instructionRegistry;
};

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//
enum class MMAOpndKind { MatrixA, MatrixB, MatrixC, MatrixD };
struct MMAInstructionInterface {
  // Get supported Matrix shapes
  virtual llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
  getSupportedShapes(Type dataType, MMAOpndKind matrixType) = 0;
  // @TODO: This method takes an context object as a parameter, this is to
  // create the Type objects from the same context. Since type objects are
  // uniqued in a specific context, to do things like "aType == bType" (where
  // aType and bType are both same type) kind of checks, the both types should
  // be from the same context.
  //
  // One alternative to this is to create enum to represent each types, but this
  // adds an extra burden to user to convert these enums to specific types. In
  // fact the utility that would convert enumToType() and vice versa would still
  // have to use the context object.
  //
  // Untill we have a better solution, we stick to passing context object to
  // this method.
  virtual llvm::SmallVector<Type, 8>
  getSupportedTypes(MLIRContext &context, MMAOpndKind matrixType) = 0;

  virtual llvm::SmallVector<uint32_t, 8> getSupportedM(Type type) const = 0;
  virtual llvm::SmallVector<uint32_t, 8> getSupportedK(Type type) const = 0;
  virtual llvm::SmallVector<uint32_t, 8> getSupportedN(Type type) const = 0;
  virtual bool isLaneLayoutRowMajorOrder() const = 0;
  virtual ~MMAInstructionInterface() = default;
};

// Interface for subgroup-level 2D block instructions (load / store / prefetch).
// All three describe the set of hardware-supported block shapes via
// (width, height, count) tuples and share a packed-format bit size. The
// transform / transpose / upConv flags are only meaningful for loads; store
// and prefetch implementations ignore them.
struct BlockIOInstructionInterface {
  using BlockShapes =
      std::tuple<llvm::ArrayRef<int>, llvm::ArrayRef<int>, llvm::ArrayRef<int>>;

  // Returns the supported (widths, heights, counts) for the given element
  // type, or std::nullopt if the element type is unsupported.
  std::optional<BlockShapes>
  getBlockWidthHeightCount(Type elemTy, bool hasTransform = false,
                           bool hasTranspose = false,
                           bool upConv = false) const {
    return computeBlockWidthHeightCount(elemTy, hasTransform, hasTranspose,
                                        upConv);
  }

  // Bit size of the packed format used by this block instruction.
  virtual int32_t getPackedFormatBitSize() const = 0;
  virtual ~BlockIOInstructionInterface() = default;

protected:
  virtual std::optional<BlockShapes>
  computeBlockWidthHeightCount(Type elemTy, bool hasTransform,
                               bool hasTranspose, bool upConv) const = 0;
};

//===----------------------------------------------------------------------===//
// Common virtual ISA instructions (shared across architectures)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SPIRV
//===----------------------------------------------------------------------===//
template <InstructionKind Kind>
struct ScatterIoInstructionInterface : public Instruction {
  static_assert(Kind == InstructionKind::LoadGather ||
                    Kind == InstructionKind::StoreScatter,
                "ScatterIO only supports LoadGather / StoreScatter");

  ScatterIoInstructionInterface() : Instruction(Kind, InstructionScope::Lane) {}

  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == Kind;
  }

  virtual int32_t getMaxLaneAccessSizeBytes() const = 0;
  virtual ~ScatterIoInstructionInterface() = default;
};
struct LoadGatherInstruction
    : public ScatterIoInstructionInterface<InstructionKind::LoadGather> {
  int32_t getMaxLaneAccessSizeBytes() const override { return 16; }
};

struct StoreScatterInstruction
    : public ScatterIoInstructionInterface<InstructionKind::StoreScatter> {
  int32_t getMaxLaneAccessSizeBytes() const override { return 16; }
};

//===----------------------------------------------------------------------===//
// SPIRV / OpenCL-extension subgroup instructions
//
// These come from cl_intel_subgroup_2d_block_io and
// cl_intel_subgroup_matrix_multiply_accumulate. A uArch only needs to
// subclass when it diverges from the extension defaults.
//===----------------------------------------------------------------------===//

struct Subgroup2DBlockStoreInstruction : public Instruction,
                                         public BlockIOInstructionInterface {
  Subgroup2DBlockStoreInstruction()
      : Instruction(InstructionKind::Subgroup2DBlockStore,
                    InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::Subgroup2DBlockStore;
  }
  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html#_add_a_new_section_5_2_x_cl_intel_subgroup_2d_block_io
  // Stores ignore the transform / transpose / upConv flags.
  int32_t getPackedFormatBitSize() const override { return 16; }

protected:
  std::optional<BlockShapes>
  computeBlockWidthHeightCount(Type elemTy, bool /*hasTransform*/,
                               bool /*hasTranspose*/,
                               bool /*upConv*/) const override {
    static const int kHeight[] = {1, 2, 4, 8};
    static const int kWidth16[] = {16};
    static const int kCount[] = {1};
    const int elemByteSize = elemTy.getIntOrFloatBitWidth() / 8;
    if (elemByteSize == 1 || elemByteSize == 2 || elemByteSize == 4)
      return std::make_tuple(llvm::ArrayRef<int>(kWidth16),
                             llvm::ArrayRef<int>(kHeight),
                             llvm::ArrayRef<int>(kCount));
    return std::nullopt;
  }
};

struct Subgroup2DBlockLoadInstruction : public Instruction,
                                        public BlockIOInstructionInterface {
  Subgroup2DBlockLoadInstruction()
      : Instruction(InstructionKind::Subgroup2DBlockLoad,
                    InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::Subgroup2DBlockLoad;
  }

  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html#_add_a_new_section_5_2_x_cl_intel_subgroup_2d_block_io
  int32_t getPackedFormatBitSize() const override { return 16; }

protected:
  std::optional<BlockShapes>
  computeBlockWidthHeightCount(Type elemTy, bool hasTransform,
                               bool hasTranspose, bool upConv) const override {
    static const int kHeightAtLeast1[] = {1, 2, 4, 8, 16, 32};
    static const int kHeightAtLeast8[] = {8, 16, 32};
    static const int kHeightAtLeast16[] = {16, 32};
    static const int kHeight32[] = {32};
    static const int kHeight64[] = {64};

    static const int kWidth64[] = {64};
    static const int kWidth32[] = {32};
    static const int kWidth16[] = {16};
    static const int kWidthAtLeast16[] = {16, 32};
    static const int kWidthAtLeast32[] = {32, 64};
    static const int kWidth8[] = {8};

    static const int32_t kCount1[] = {1};
    static const int32_t kCount2[] = {1, 2};
    static const int32_t kCount4[] = {1, 2, 4};
    static const int32_t kCount4Only[] = {4};
    // (elemBits, transform, transpose, upConvert)
    using Key = std::tuple<int, uint8_t, uint8_t, uint8_t>;
    // (widths, heights, counts)
    using Value = std::tuple<llvm::ArrayRef<int32_t>, llvm::ArrayRef<int32_t>,
                             llvm::ArrayRef<int32_t>>;
    // The table is keyed on element bit width so sub-byte elements can be
    // expressed directly. 4-bit elements are packed two-per-byte, so their
    // widths (or heights, when transformed) are double the 8-bit rows.
    static const llvm::DenseMap<Key, Value> kMap = {
        {{8, false, false, false}, {kWidthAtLeast16, kHeightAtLeast1, kCount2}},
        {{8, false, false, true}, {kWidth16, kHeightAtLeast8, kCount4Only}},
        {{16, false, false, false}, {kWidth16, kHeightAtLeast1, kCount2}},
        {{32, false, false, false}, {kWidth16, kHeightAtLeast1, kCount1}},
        // Block Loads with Transform:
        {{8, true, false, false}, {kWidth16, kHeight32, kCount4}},
        {{16, true, false, false}, {kWidth16, kHeightAtLeast16, kCount2}},
        // Block Loads with Transpose:
        {{8, false, true, false}, {kWidth32, kHeightAtLeast16, kCount1}},
        {{16, false, true, false}, {kWidth16, kHeightAtLeast16, kCount1}},
        {{32, false, true, false}, {kWidth8, kHeightAtLeast16, kCount1}},
        // 4-bit elements (sub-byte):
        {{4, false, false, false}, {kWidthAtLeast32, kHeightAtLeast1, kCount2}},
        {{4, false, false, true}, {kWidth32, kHeightAtLeast8, kCount4Only}},
        {{4, true, false, false}, {kWidth16, kHeight64, kCount4}},
        {{4, false, true, false}, {kWidth64, kHeightAtLeast16, kCount1}}};
    int elemBitSize = elemTy.getIntOrFloatBitWidth();
    auto it = kMap.find({elemBitSize, hasTransform, hasTranspose, upConv});
    if (it != kMap.end())
      return it->second;
    return std::nullopt;
  }
};

struct Subgroup2DBlockPrefetchInstruction : public Instruction,
                                            public BlockIOInstructionInterface {
  Subgroup2DBlockPrefetchInstruction()
      : Instruction(InstructionKind::Subgroup2DBlockPrefetch,
                    InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::Subgroup2DBlockPrefetch;
  }
  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_buffer_prefetch.html#_add_a_new_section_6_15_x_sub_group_prefetch_functions
  // Prefetches ignore the transform / transpose / upConv flags.
  int32_t getPackedFormatBitSize() const override { return 16; }

protected:
  std::optional<BlockShapes>
  computeBlockWidthHeightCount(Type elemTy, bool /*hasTransform*/,
                               bool /*hasTranspose*/,
                               bool /*upConv*/) const override {
    static const int kHeightAtLeast1[] = {1, 2, 4, 8, 16, 32};

    static const int kWidth32[] = {32};
    static const int kWidth16[] = {16};

    static const int32_t kCount1[] = {1};
    static const int32_t kCount2[] = {1, 2};
    // elemBytes
    using Key = int;
    // (widths, heights, counts)
    using Value = std::tuple<llvm::ArrayRef<int32_t>, llvm::ArrayRef<int32_t>,
                             llvm::ArrayRef<int32_t>>;
    static const llvm::DenseMap<Key, Value> kMap = {
        {1, {kWidth32, kHeightAtLeast1, kCount2}},
        {2, {kWidth16, kHeightAtLeast1, kCount2}},
        {4, {kWidth16, kHeightAtLeast1, kCount1}},
    };
    const int elemByteSize = elemTy.getIntOrFloatBitWidth() / 8;
    auto it = kMap.find(elemByteSize);
    if (it != kMap.end())
      return it->second;
    return std::nullopt;
  }
};

struct SubgroupMatrixMultiplyAcc : public Instruction,
                                   public MMAInstructionInterface {
  SubgroupMatrixMultiplyAcc(unsigned packedFormatBitSizeA,
                            unsigned packedFormatBitSizeB)
      : Instruction(InstructionKind::SubgroupMatrixMultiplyAcc,
                    InstructionScope::Subgroup),
        packedFormatBitSizeA(packedFormatBitSizeA),
        packedFormatBitSizeB(packedFormatBitSizeB) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() ==
           InstructionKind::SubgroupMatrixMultiplyAcc;
  }
  // Source:
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html

  llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
  getSupportedShapes(Type dataType, MMAOpndKind matrixType) override;
  llvm::SmallVector<Type, 8> getSupportedTypes(MLIRContext &context,
                                               MMAOpndKind matrixType) override;

  llvm::SmallVector<uint32_t, 8> getSupportedM(Type type) const override;
  llvm::SmallVector<uint32_t, 8> getSupportedK(Type type) const override;
  llvm::SmallVector<uint32_t, 8> getSupportedN(Type type) const override;

  unsigned getPackedFormatBitSizeA() const { return packedFormatBitSizeA; }
  unsigned getPackedFormatBitSizeB() const { return packedFormatBitSizeB; }
  bool isLaneLayoutRowMajorOrder() const override { return true; }

protected:
  const unsigned packedFormatBitSizeA;
  const unsigned packedFormatBitSizeB;
};

struct SubgroupScaledMatrixMultiplyAcc : public Instruction,
                                         public MMAInstructionInterface {
  SubgroupScaledMatrixMultiplyAcc(unsigned packedFormatBitSizeA,
                                  unsigned packedFormatBitSizeB)
      : Instruction(InstructionKind::SubgroupScaledMatrixMultiplyAcc,
                    InstructionScope::Subgroup),
        packedFormatBitSizeA(packedFormatBitSizeA),
        packedFormatBitSizeB(packedFormatBitSizeB) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() ==
           InstructionKind::SubgroupScaledMatrixMultiplyAcc;
  }
  // Source:
  // https://github.com/intel/llvm/blob/sycl/sycl/doc/design/spirv-extensions/SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate.asciidoc

  llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
  getSupportedShapes(Type dataType, MMAOpndKind matrixType) override;
  llvm::SmallVector<Type, 8> getSupportedTypes(MLIRContext &context,
                                               MMAOpndKind matrixType) override;

  llvm::SmallVector<uint32_t, 8> getSupportedM(Type type) const override;
  llvm::SmallVector<uint32_t, 8> getSupportedK(Type type) const override;
  llvm::SmallVector<uint32_t, 8> getSupportedN(Type type) const override;

  unsigned getPackedFormatBitSizeA() const { return packedFormatBitSizeA; }
  unsigned getPackedFormatBitSizeB() const { return packedFormatBitSizeB; }
  bool isLaneLayoutRowMajorOrder() const override { return true; }

protected:
  const unsigned packedFormatBitSizeA;
  const unsigned packedFormatBitSizeB;
};

//===----------------------------------------------------------------------===//
// Inline implementations
//===----------------------------------------------------------------------===//

namespace util {
inline llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
crossProduct(const llvm::SmallVector<uint32_t, 8> &a,
             const llvm::SmallVector<uint32_t, 8> &b) {
  llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16> result;
  for (unsigned x : a)
    for (unsigned y : b)
      result.emplace_back(x, y);
  return result;
}
} // namespace util

inline llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
SubgroupMatrixMultiplyAcc::getSupportedShapes(Type dataType,
                                              MMAOpndKind matrixType) {
  auto M = getSupportedM(dataType);
  auto K = getSupportedK(dataType);
  auto N = getSupportedN(dataType);
  switch (matrixType) {
  case MMAOpndKind::MatrixA:
    return util::crossProduct(M, K);
  case MMAOpndKind::MatrixB:
    return util::crossProduct(K, N);
  case MMAOpndKind::MatrixC:
  case MMAOpndKind::MatrixD:
    return util::crossProduct(M, N);
  }
  return {};
}

inline llvm::SmallVector<Type, 8>
SubgroupMatrixMultiplyAcc::getSupportedTypes(MLIRContext &context,
                                             MMAOpndKind matrixType) {
  Type bf16Type = BFloat16Type::get(&context);
  Type f16Type = Float16Type::get(&context);
  Type tf32Type = FloatTF32Type::get(&context);
  Type f32Type = Float32Type::get(&context);

  switch (matrixType) {
  case MMAOpndKind::MatrixA:
  case MMAOpndKind::MatrixB:
    return {bf16Type, f16Type, tf32Type};
  case MMAOpndKind::MatrixC:
  case MMAOpndKind::MatrixD:
    return {bf16Type, f16Type, f32Type};
  }
  return {};
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupMatrixMultiplyAcc::getSupportedM(Type type) const {
  return {1, 2, 3, 4, 5, 6, 7, 8};
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupMatrixMultiplyAcc::getSupportedK(Type type) const {
  assert(type.isIntOrFloat() && "Matrix type must be int or float");
  auto bitWidth = type.getIntOrFloatBitWidth();
  uint32_t kSize = 0;
  switch (bitWidth) {
  case 4:
    kSize = 64;
    break;
  case 8:
    kSize = 32;
    break;
  case 16:
    kSize = 16;
    break;
  case 32:
    kSize = 8;
    break;
  default:
    llvm_unreachable("Invalid int or float");
  }
  return {kSize};
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupMatrixMultiplyAcc::getSupportedN(Type type) const {
  return {16};
}

inline llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
SubgroupScaledMatrixMultiplyAcc::getSupportedShapes(Type dataType,
                                                    MMAOpndKind matrixType) {
  // Avoid calling getSupportedK for C/D types (which are f32/bf16
  // and not valid for the K-dimension bit-width calculation).
  switch (matrixType) {
  case MMAOpndKind::MatrixA:
    return util::crossProduct(getSupportedM(dataType), getSupportedK(dataType));
  case MMAOpndKind::MatrixB:
    return util::crossProduct(getSupportedK(dataType), getSupportedN(dataType));
  case MMAOpndKind::MatrixC:
  case MMAOpndKind::MatrixD:
    return util::crossProduct(getSupportedM(dataType), getSupportedN(dataType));
  }
  return {};
}

inline llvm::SmallVector<Type, 8>
SubgroupScaledMatrixMultiplyAcc::getSupportedTypes(MLIRContext &context,
                                                   MMAOpndKind matrixType) {
  Type f8E4M3FNType = Float8E4M3FNType::get(&context);
  Type f8E5M2Type = Float8E5M2Type::get(&context);
  Type f4E2M1FNType = Float4E2M1FNType::get(&context);
  Type bf16Type = BFloat16Type::get(&context);
  Type f32Type = Float32Type::get(&context);

  switch (matrixType) {
  case MMAOpndKind::MatrixA:
  case MMAOpndKind::MatrixB:
    return {f8E4M3FNType, f8E5M2Type, f4E2M1FNType};
  case MMAOpndKind::MatrixC:
  case MMAOpndKind::MatrixD:
    return {bf16Type, f32Type};
  }
  return {};
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupScaledMatrixMultiplyAcc::getSupportedM(Type type) const {
  return {8};
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupScaledMatrixMultiplyAcc::getSupportedK(Type type) const {
  assert(type.isIntOrFloat() && "Matrix type must be int or float");
  auto bitWidth = type.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 4:
    return {64}; // FP4: scale K by 4 (base 16-bit K=16 -> 64)
  case 8:
    return {32}; // FP8: scale K by 2 (base 16-bit K=16 -> 32)
  default:
    // Scaled dpas only supports FP8 (8-bit) and FP4 (4-bit) types for A/B
    // matrices. Return empty so callers can gracefully reject unsupported
    // types instead of aborting.
    return {};
  }
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupScaledMatrixMultiplyAcc::getSupportedN(Type type) const {
  return {16};
}

} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UARCH_UARCHBASE_H
