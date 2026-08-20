//===-- PISARegisterInfo.td - PISA Register defs ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let Namespace = "PISA" in {

  class PISARegisterClass<list<ValueType> regTypes, int alignment, dag regList>
      : RegisterClass<"PISA", regTypes, alignment, regList> {}

  // .reg style registers. See the PISA specification for register variables:
  // https://intel.github.io/pisa/variables.html#register-variables
  class GeneralRegisterClass<list<ValueType> regTypes, int alignment,
                             dag regList>
      : PISARegisterClass<regTypes, alignment, regList> {}

  // .pred style registers. See the PISA specification for predicate variables:
  // https://intel.github.io/pisa/variables.html#predicate-variables
  class PredRegisterClass<list<ValueType> regTypes, int alignment, dag regList>
      : PISARegisterClass<regTypes, alignment, regList> {}

  def DummyReg : Register<"%null">;

  foreach Index = 0-63 in {
    def Reg32b_#Index : Register<"">;
  }
  foreach Index = 0-3 in {
    def Reg64b_#Index : Register<"">;
    def Pred_#Index : Register<"">;
  }

  def Pred : PredRegisterClass<[i1], 8,
    (add DummyReg, (sequence "Pred_%u", 0, 3))>;
  def Reg32b : GeneralRegisterClass<[i32, f32], 32,
    (add DummyReg, (sequence "Reg32b_%u", 0, 63))>;
  def Reg64b : GeneralRegisterClass<[i64, f64], 64,
    (add (sequence "Reg64b_%u", 0, 3))>;
}
