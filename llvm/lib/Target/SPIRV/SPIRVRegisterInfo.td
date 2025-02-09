//===-- SPIRVRegisterInfo.td - SPIR-V Register defs --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Declarations that describe the SPIR-V register file.
//
//===----------------------------------------------------------------------===//

let Namespace = "SPIRV" in {
  // Pointer types for patterns with the GlobalISelEmitter
  def p64 : PtrValueType <i64, 0>;

  class VTPtrVec<int nelem, PtrValueType ptr>
      : VTVec<nelem, ValueType<ptr.Size, ptr.Value>, ptr.Value> {
    int isPointer = true;
  }

  def v2p64 : VTPtrVec<2, p64>;

  // Class for type registers
  def TYPE0 : Register<"TYPE0">;
  def TYPE : RegisterClass<"SPIRV", [i64], 64, (add TYPE0)>;

  // Class for non-type registers
  def ID0 : Register<"ID0">;
  def fID0 : Register<"fID0">;
  def pID0 : Register<"pID0">;
  def vID0 : Register<"vID0">;
  def vfID0 : Register<"vfID0">;
  def vpID0 : Register<"vpID0">;

  def iID : RegisterClass<"SPIRV", [i64], 64, (add ID0)>;
  def fID : RegisterClass<"SPIRV", [f64], 64, (add fID0)>;
  def pID : RegisterClass<"SPIRV", [p64], 64, (add pID0)>;
  def vID : RegisterClass<"SPIRV", [v2i64], 64, (add vID0)>;
  def vfID : RegisterClass<"SPIRV", [v2f64], 64, (add vfID0)>;
  def vpID : RegisterClass<"SPIRV", [v2p64], 64, (add vpID0)>;
  
  def ID : RegisterClass<
      "SPIRV",
      [i64, f64, p64, v2i64, v2f64, v2p64],
      64,
      (add iID, fID, pID, vID, vfID, vpID)>;

  // A few instructions like OpName can take ids from both type and non-type
  // instructions, so we need a super-class to allow for both to count as valid
  // arguments for these instructions.
  def ANY : RegisterClass<"SPIRV", [i64], 64, (add TYPE, ID)>;
}
