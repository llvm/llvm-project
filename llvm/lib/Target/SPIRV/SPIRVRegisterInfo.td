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
  def p0 : PtrValueType <i32, 0>;

  class P0Vec<ValueType scalar>
      : PtrValueType <scalar, 0> {
    let nElem = 2;
    let ElementType = p0;
    let isInteger = false;
    let isFP = false;
    let isVector = true;
  }

  def v2p0 : P0Vec<i32>;
  // All registers are for 32-bit identifiers, so have a single dummy register

  // Class for registers that are the result of OpTypeXXX instructions
  def TYPE0 : Register<"TYPE0">;
  def TYPE : RegisterClass<"SPIRV", [i32], 32, (add TYPE0)>;

  // Class for every other non-type ID
  def ID0 : Register<"ID0">;
  def ID : RegisterClass<"SPIRV", [i32], 32, (add ID0)>;
  def fID0 : Register<"fID0">;
  def fID : RegisterClass<"SPIRV", [f32], 32, (add fID0)>;
  def pID0 : Register<"pID0">;
  def pID : RegisterClass<"SPIRV", [p0], 32, (add pID0)>;
  def vID0 : Register<"vID0">;
  def vID : RegisterClass<"SPIRV", [v2i32], 32, (add vID0)>;
  def vfID0 : Register<"vfID0">;
  def vfID : RegisterClass<"SPIRV", [v2f32], 32, (add vfID0)>;
  def vpID0 : Register<"vpID0">;
  def vpID : RegisterClass<"SPIRV", [v2p0], 32, (add vpID0)>;

  def ANYID : RegisterClass<"SPIRV", [i32, f32, p0, v2i32, v2f32], 32, (add ID, fID, pID, vID, vfID)>;

  // A few instructions like OpName can take ids from both type and non-type
  // instructions, so we need a super-class to allow for both to count as valid
  // arguments for these instructions.
  def ANY : RegisterClass<"SPIRV", [i32], 32, (add TYPE, ID)>;
}
