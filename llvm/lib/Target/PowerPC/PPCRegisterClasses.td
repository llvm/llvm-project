//===-- PPCRegisterClasses.td - PPC Register Class Definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base classes for PowerPC register classes to reduce
// repetition and make it easier to define new register classes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base Register Class Definitions
//===----------------------------------------------------------------------===//

// Base class for all PPC register classes - sets namespace to "PPC"
class PPCRegisterClass<list<ValueType> regTypes, int alignment, dag regList>
  : RegisterClass<"PPC", regTypes, alignment, regList>;

//===----------------------------------------------------------------------===//
// Classes for Generating RegisterOperand for Existing RegisterClass
//===----------------------------------------------------------------------===//

// Base class that creates a RegisterOperand for an already-defined RegisterClass,
// assuming the AsmOperandClass already exists with the standard naming convention
// (PPCReg<regClassName>AsmOperand).
//
// Usage: def spe4rc : PPCRegOperandOnly<"GPRC">;
//   Creates: spe4rc RegisterOperand wrapping GPRC, using PPCRegGPRCAsmOperand
class PPCRegOperandOnly<string regClassName>
  : RegisterOperand<!cast<RegisterClass>(regClassName)> {
  let ParserMatchClass = !cast<AsmOperandClass>(!strconcat("PPCReg", regClassName, "AsmOperand"));
}

// Multiclass that generates both the AsmOperandClass and RegisterOperand for an
// already-defined RegisterClass. This eliminates the repetitive pattern of
// manually defining these for each register class.
//
// Usage examples:
//   defm GPRC : PPCRegOperand<"isRegNumber">;
//     Creates: PPCRegGPRCAsmOperand and gprc (lowercase of GPRC)
//   defm FpRC : PPCRegOperand<"isEvenRegNumber", "fpairrc">;
//     Creates: PPCRegFpRCAsmOperand and fpairrc (custom name)
//
// The NAME in the defm statement should match the RegisterClass name.

multiclass PPCRegOperand<string predicate, string operandName = ""> {
  // Define the AsmOperandClass
  def "PPCReg"#NAME#"AsmOperand" : AsmOperandClass {
    let Name = "Reg"#NAME;
    let PredicateMethod = predicate;
  }
  
  // Define the RegisterOperand
  // Use custom name if provided, otherwise use lowercase of NAME
  def !if(!eq(operandName, ""), !tolower(NAME), operandName)
    : RegisterOperand<!cast<RegisterClass>(NAME)> {
    let ParserMatchClass = !cast<AsmOperandClass>("PPCReg"#NAME#"AsmOperand");
  }
}

//===----------------------------------------------------------------------===//
// Variant Register Class Definitions
//===----------------------------------------------------------------------===//

// Register class that is not allocatable
class PPCNonAllocatableRegisterClass<list<ValueType> regTypes, int alignment,
                                     dag regList>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let isAllocatable = 0;
}

// Register class with explicit size
class PPCRegisterClassWithSize<list<ValueType> regTypes, int alignment,
                               dag regList, int size>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let Size = size;
}

// Register class with allocation priority and size
class PPCRegisterClassWithPriority<list<ValueType> regTypes, int alignment,
                                   dag regList, int allocPriority,
                                   bit globalPriority, int size>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let AllocationPriority = allocPriority;
  let GlobalPriority = globalPriority;
  let Size = size;
}

//===----------------------------------------------------------------------===//
// Specialized Base Classes for Common PPC Patterns
//===----------------------------------------------------------------------===//

// GPR-style register class with alternative orders for different ABIs
// Merged PPCRegisterClassWithAltOrders into this class since it was only used here
class PPCGPRRegisterClass<list<ValueType> regTypes, int alignment, dag regList,
                          dag altOrder1, dag altOrder2>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let AltOrders = [altOrder1, altOrder2];
  let AltOrderSelect = [{
    return MF.getSubtarget<PPCSubtarget>().getGPRAllocationOrderIdx();
  }];
}

// Condition register class with alternative orders for ELFv2 ABI
class PPCCRRegisterClass<list<ValueType> regTypes, int alignment, dag regList,
                         dag altOrder, int size = 0>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let AltOrders = [altOrder];
  let AltOrderSelect = [{
    return MF.getSubtarget<PPCSubtarget>().isELFv2ABI() &&
           MF.getInfo<PPCFunctionInfo>()->isNonVolatileCRDisabled();
  }];
  let Size = !if(!ne(size, 0), size, 0);
}

// Paired register class with alternative orders for 64-bit ELF ABI
class PPCPairedRegisterClass<list<ValueType> regTypes, int alignment,
                             dag regList, dag altOrder, int size>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let AltOrders = [altOrder];
  let AltOrderSelect = [{
    return MF.getSubtarget<PPCSubtarget>().is64BitELFABI();
  }];
  let Size = size;
}

