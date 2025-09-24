//===--- uArch.h ------------------------------------------------*- C++ -*-===//
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

namespace mlir {
namespace xegpu {
namespace uArch {

// An enum class to represent the scope of an instruction
enum class InstructionScope { WorkItem, Subgroup, Workgroup, Cluster };

enum class InstructionName {
  DPAS, // Dot Product Accumulate Systolic (DPAS) is a matrix multiply-add
        // operation
  // Add more instructions as needed
};

// A struct to represent basic information about an instruction
// This struct is used to represent the information about an instruction in the
// uArch The information includes:
// - the name of the instruction,
// - the description of the instruction
// - the scope of the instruction,
//
// The information is represented as strings
// For example, the information about an instruction can be represented as:
// Instruction instr = {"dpas", "Dot Product Accumulate Systolic  (DPAS) is a
// matrix multiply-add operation", "subgroup"};

// The primary purpose of the Instruction struct is to provide a generic way to
// represent information about an instruction and to use this information to
// generate the uArch. Specifc instruction in a uArch can inherit from this
// struct and add more fields as needed

struct Instruction {
  // @TODO: Add more fields as needed
  Instruction(std::string name, std::string desc)
      : name(std::move(name)), description(std::move(desc)) {}

  virtual ~Instruction() = default;
  // Get methods
  std::string getName() { return name; }
  std::string getDescription() { return description; }
  InstructionScope getScope() { return scope; }

protected:
  std::string name;
  std::string description;
  InstructionScope scope;
};

enum class RegisterFileMode : uint8_t { Small, Large };
enum class RegisterFileType : uint8_t { GRF, ARF };

// A struct to represent register file information
struct RegisterFileInfo {
  // Constructor
  RegisterFileInfo() = default;
  RegisterFileInfo(uint32_t size, const std::vector<RegisterFileMode> &mode,
                   const std::vector<uint32_t> &numRegs)
      : size(size), mode(mode), numRegsPerThreadPerMode(numRegs) {}

  uint32_t getSize() const { return size; }
  const std::vector<RegisterFileMode> &getModes() const { return mode; }
  const std::vector<uint32_t> &getNumRegsPerThreadPerMode() const {
    return numRegsPerThreadPerMode;
  }

protected:
  uint32_t size;                      // size per register in bits
  std::vector<RegisterFileMode> mode; // e.g., "small", "large" GRF modes
  std::vector<uint32_t>
      numRegsPerThreadPerMode; // number of registers per thread per mode
  // TODO: Add more fields as needed (e.g., num_banks, bank_size, num_ports,
  // port_width, bank_conflicts)
};

enum class CacheHierarchyLevel { L1 = 1, L2 = 2, L3 = 3 };
// A struct to represent cache information
struct CacheInfo {
  // Constructor
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

// A struct to represent the uArch
// This struct is used to represent the microarchitecture of a target device
// The uArch includes:
// - the name of the uArch,
// - the description of the uArch,
// - uArch hierarchy
// - Rgister File information
// - Cache information
// - the set of instructions supported by the uArch,
struct uArch {
  // Constructor
  uArch() = default;
  uArch(const std::string &name, const std::string &description,
        const std::map<RegisterFileType, RegisterFileInfo> &register_file_info =
            {},
        const std::vector<CacheInfo> &cache_info = {},
        const std::map<std::string, std::shared_ptr<Instruction>>
            &instructions = {})
      : name(name), description(description),
        registerFileInfo(register_file_info), cacheInfo(cache_info),
        instructions(instructions) {}

  // Get methods
  const std::string &getName() const { return name; }

  const std::string &getDescription() const { return description; }

  const std::map<RegisterFileType, RegisterFileInfo> &
  getRegisterFileInfo() const {
    return registerFileInfo;
  }

  const std::vector<CacheInfo> &getCacheInfo() const { return cacheInfo; }

  const std::map<std::string, std::shared_ptr<Instruction>> &
  getInstructions() const {
    return instructions;
  }

  // Get the name of the supported instruction names for that
  // architecture. It returns the names of the instructions added to the uArch.
  std::vector<std::string> getSupportedInstructionNames() const {
    std::vector<std::string> instructionNames;
    for (const auto &inst : instructions) {
      instructionNames.push_back(inst.first);
    }
    return instructionNames;
  }

  // Checks if an instruction is supported in this uArch
  bool checkSupportedInstruction(const std::string &instructionName) const {
    return instructions.find(instructionName) != instructions.end();
  }

protected:
  std::string name; // Similar to target triple
  std::string description;
  std::map<RegisterFileType, RegisterFileInfo> registerFileInfo;
  std::vector<CacheInfo> cacheInfo;
  std::map<std::string, std::shared_ptr<Instruction>> instructions;
};

// A struct to represent shared memory information
struct SharedMemory {
  // Constructor
  SharedMemory(uint32_t size, uint32_t alignment)
      : size(size), alignment(alignment) {}

  // Getters
  uint32_t getSize() const { return size; }
  uint32_t getAlignment() const { return alignment; }

protected:
  uint32_t size;      // in bytes
  uint32_t alignment; // in bytes
  // @TODO: Add more fields as needed (e.g., latency, throughput, bandwidth)
};

struct uArchMap {
public:
  // Singleton instance
  static uArchMap &instance() {
    static uArchMap instance;
    return instance;
  }

  // Insert or update a key-value pair
  void insert(const std::string &key, std::shared_ptr<uArch> value) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    // map_[key] = std::move(value); // safe to overwrite
    map_.emplace(key, std::move(value)); // safe to overwrite
  }

  // Get a value by key (concurrent safe read)
  std::shared_ptr<uArch> get(const std::string &key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it != map_.end())
      return it->second;
    return nullptr;
  }

  // Check if a key exists
  bool contains(const std::string &key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return map_.find(key) != map_.end();
  }

  // Remove a key
  bool erase(const std::string &key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    return map_.erase(key) > 0;
  }

private:
  uArchMap() = default;
  uArchMap(const uArchMap &) = delete;
  uArchMap &operator=(const uArchMap &) = delete;

  mutable std::shared_mutex mutex_;
  std::map<std::string, std::shared_ptr<uArch>> map_;
};

} // namespace uArch
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UARCH_UARCHBASE_H
