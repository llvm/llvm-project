//===- xray-sleds.h - XRay Sleds Data Structure ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the structure used to represent XRay instrumentation map entries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_XRAY_XRAY_SLEDS_H
#define LLVM_TOOLS_LLVM_XRAY_XRAY_SLEDS_H

namespace llvm {
namespace xray {

struct SledEntry {
  enum class FunctionKinds { ENTRY, EXIT, TAIL };

  uint64_t Address;
  uint64_t Function;
  FunctionKinds Kind;
  bool AlwaysInstrument;
};

} // namespace xray
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_XRAY_XRAY_SLEDS_H
