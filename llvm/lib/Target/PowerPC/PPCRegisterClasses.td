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

// Register class with explicit size
class PPCRegisterClassWithSize<list<ValueType> regTypes, int alignment,
                               dag regList, int size>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let Size = size;
}

// Register class that is not allocatable
class PPCNonAllocatableRegisterClass<list<ValueType> regTypes, int alignment,
                                     dag regList>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let isAllocatable = 0;
}

// Register class with custom copy cost
class PPCRegisterClassWithCopyCost<list<ValueType> regTypes, int alignment,
                                   dag regList, int copyCost>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let CopyCost = copyCost;
}

// Register class with allocation priority
class PPCRegisterClassWithPriority<list<ValueType> regTypes, int alignment,
                                   dag regList, int allocPriority,
                                   bit globalPriority = 0, int size = 0>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let AllocationPriority = allocPriority;
  let GlobalPriority = globalPriority;
  let Size = size;
}

// Register class with alternative allocation orders
class PPCRegisterClassWithAltOrders<list<ValueType> regTypes, int alignment,
                                    dag regList, list<dag> altOrders,
                                    code altOrderSelect>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let AltOrders = altOrders;
  let AltOrderSelect = altOrderSelect;
}

// Register class with alternative orders and size
class PPCRegisterClassWithAltOrdersAndSize<list<ValueType> regTypes, int alignment,
                                           dag regList, list<dag> altOrders,
                                           code altOrderSelect, int size>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let AltOrders = altOrders;
  let AltOrderSelect = altOrderSelect;
  let Size = size;
}

// Non-allocatable register class with copy cost
class PPCNonAllocatableRegisterClassWithCopyCost<list<ValueType> regTypes,
                                                 int alignment, dag regList,
                                                 int copyCost>
  : PPCRegisterClass<regTypes, alignment, regList> {
  let isAllocatable = 0;
  let CopyCost = copyCost;
}

//===----------------------------------------------------------------------===//
// Specialized Base Classes for Common PPC Patterns
//===----------------------------------------------------------------------===//

// GPR-style register class with alternative orders for different ABIs
class PPCGPRRegisterClass<list<ValueType> regTypes, int alignment, dag regList,
                          dag altOrder1, dag altOrder2>
  : PPCRegisterClassWithAltOrders<regTypes, alignment, regList,
      [altOrder1, altOrder2],
      [{
        return MF.getSubtarget<PPCSubtarget>().getGPRAllocationOrderIdx();
      }]>;

// Vector/VSX register class with size
class PPCVectorRegisterClass<list<ValueType> regTypes, int alignment,
                             dag regList, int size>
  : PPCRegisterClassWithSize<regTypes, alignment, regList, size>;

// Accumulator register class with high allocation priority
class PPCAccumulatorRegisterClass<list<ValueType> regTypes, int alignment,
                                  dag regList, int allocPriority,
                                  bit globalPriority, int size>
  : PPCRegisterClassWithPriority<regTypes, alignment, regList,
                                 allocPriority, globalPriority, size>;

// DMR-style register class with size
class PPCDMRRegisterClass<list<ValueType> regTypes, int alignment,
                          dag regList, int size>
  : PPCRegisterClassWithSize<regTypes, alignment, regList, size>;

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

// Special purpose register class (non-allocatable)
class PPCSPRegisterClass<list<ValueType> regTypes, int alignment, dag regList>
  : PPCNonAllocatableRegisterClass<regTypes, alignment, regList>;

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