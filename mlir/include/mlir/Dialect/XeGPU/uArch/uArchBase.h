//===- uArch.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Base uArch definition for different architectures.
//
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UARCH_UARCHBASE_H
#define MLIR_DIALECT_XEGPU_UARCH_UARCHBASE_H

#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <tuple>

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace xegpu {
namespace uArch {

constexpr unsigned generalPackedFormatBitSize{32};

// An enum class to represent the scope of an instruction
enum class InstructionScope { Lane, Subgroup, Workgroup, Cluster };
enum class InstructionKind {
  SubgroupMatrixMultiplyAcc, // Dot Product Accumulate Systolic (DPAS) is a
                             // matrix multiply-add operation
  Subgroup2DBlockStore,      // Subgroup-level 2D block write instruction
  Subgroup2DBlockLoad,       // Subgroup-level 2D block load instruction
  Subgroup2DBlockPrefetch    // Subgroup-level 2D block prefetch instruction
  // @TODO: Add more instructions as needed
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
    case InstructionKind::Subgroup2DBlockStore:
      return "store_nd";
    case InstructionKind::Subgroup2DBlockLoad:
      return "load_nd";
    case InstructionKind::Subgroup2DBlockPrefetch:
      return "prefetch_nd";
    }
    llvm_unreachable("Unknown InstructionKind");
  }

  static std::optional<InstructionKind>
  parseInstructionKind(llvm::StringRef str) {
    if (str.equals_insensitive("dpas"))
      return InstructionKind::SubgroupMatrixMultiplyAcc;
    return std::nullopt;
  }

protected:
  const InstructionKind instKind; // Specific InstructionKind (e.g., DPAS)
  const InstructionScope scope;   // scope of the instruction (e.g., lane,
                                  // subgroup, workgroup, cluster)
  // @TODO: Add more fields as needed
};

enum class RegisterFileMode : uint8_t { Small, Large };
enum class RegisterFileType : uint8_t { GRF, ARF };

// A struct to represent register file information
struct RegisterFileInfo {
  // Constructor
  RegisterFileInfo() = default;
  RegisterFileInfo(uint32_t size,
                   const llvm::SmallVector<RegisterFileMode, 4> &mode,
                   const llvm::SmallVector<uint32_t, 4> &numRegs)
      : size(size), mode(mode), numRegsPerThreadPerMode(numRegs) {}

  // Get methods
  uint32_t getSize() const { return size; }

  const llvm::SmallVector<RegisterFileMode, 4> &getModes() const {
    return mode;
  }

  const llvm::SmallVector<uint32_t, 4> &getNumRegsPerThreadPerMode() const {
    return numRegsPerThreadPerMode;
  }

protected:
  uint32_t size; // size per register in bits
  llvm::SmallVector<RegisterFileMode, 4>
      mode; // e.g., "small", "large" GRF modes
  llvm::SmallVector<uint32_t, 4>
      numRegsPerThreadPerMode; // number of registers per thread per mode
};

enum class CacheHierarchyLevel { L1 = 1, L2 = 2, L3 = 3 };

// A struct to represent cache information
struct CacheInfo {
  // Constructor
  CacheInfo() = default;
  CacheInfo(uint32_t size, uint32_t line_size,
            CacheHierarchyLevel hierarchy_level)
      : size(size), line_size(line_size), hierarchy_level(hierarchy_level) {}

  virtual ~CacheInfo() = default;

  // Get methods
  uint32_t getSize() const { return size; }
  uint32_t getLineSize() const { return line_size; }
  CacheHierarchyLevel getHierarchyLevel() const { return hierarchy_level; }

protected:
  uint32_t size;
  uint32_t line_size;
  CacheHierarchyLevel hierarchy_level;
  // @TODO: Add more fields as needed (e.g., associativity, num_banks,
  // bank_size, num_ports, port_width, bank_conflicts, hierarchy_level,
  // latency, throughput, bandwidth)
};

struct uArch {
  // Constructor
  uArch(StringRef name, StringRef description,
        llvm::ArrayRef<const Instruction *> instructionRegistry)
      : name(name), description(description) {
    for (const Instruction *instr : instructionRegistry)
      this->instructionRegistry[instr->getInstructionKind()] = instr;
  }
  virtual ~uArch() = default;
  StringRef getName() const { return name; }
  StringRef getDescription() const { return description; }
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
  StringRef name;
  StringRef description;
  llvm::SmallDenseMap<InstructionKind, const Instruction *, 32>
      instructionRegistry;
};

// A struct to represent shared memory information
struct SharedMemory {
  // Constructor
  SharedMemory(uint32_t size, uint32_t alignment)
      : size(size), alignment(alignment) {}

  // Get methods
  uint32_t getSize() const { return size; }
  uint32_t getAlignment() const { return alignment; }

protected:
  uint32_t size;      // in bytes
  uint32_t alignment; // in bytes
  // @TODO: Add more fields as needed (e.g., latency, throughput, bandwidth)
};

struct XeCoreInfo {
  uint32_t num_threads;
  SharedMemory shared_memory;
  uint32_t num_vector_units;
  uint32_t num_matrix_units;

  XeCoreInfo(uint32_t num_threads, const SharedMemory &shared_memory,
             uint32_t num_vector_units, uint32_t num_matrix_units)
      : num_threads(num_threads), shared_memory(shared_memory),
        num_vector_units(num_vector_units), num_matrix_units(num_matrix_units) {
  }
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
  virtual bool
  checkSupportedShapesAndTypes(std::pair<uint32_t, uint32_t> AShape,
                               std::pair<uint32_t, uint32_t> BShape,
                               std::pair<uint32_t, uint32_t> CShape,
                               std::pair<uint32_t, uint32_t> DShape, Type AType,
                               Type BType, Type CType, Type DType) = 0;
  virtual bool checkSupportedTypes(Type AType, Type BType, Type CType,
                                   Type DType) = 0;
  virtual bool validate(std::pair<uint32_t, uint32_t> AShape,
                        std::pair<uint32_t, uint32_t> BShape,
                        std::pair<uint32_t, uint32_t> CShape,
                        std::pair<uint32_t, uint32_t> DShape, Type AType,
                        Type BType, Type CType, Type DType) = 0;
  virtual llvm::SmallVector<uint32_t, 8> getSupportedM(Type type) const = 0;
  virtual llvm::SmallVector<uint32_t, 8> getSupportedK(Type type) const = 0;
  virtual llvm::SmallVector<uint32_t, 8> getSupportedN(Type type) const = 0;

  virtual ~MMAInstructionInterface() = default;
};

} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UARCH_UARCHBASE_H
