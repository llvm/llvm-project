//===-- TestInterfaces.td - Test dialect interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_INTERFACES
#define TEST_INTERFACES

include "mlir/IR/OpBase.td"

// A type interface used to test the ODS generation of type interfaces.
def TestTypeInterface : TypeInterface<"TestTypeInterface"> {
  let methods = [
    InterfaceMethod<"Prints the type name.",
      "void", "printTypeA", (ins "Location":$loc), [{
        emitRemark(loc) << $_type << " - TestA";
      }]
    >,
    InterfaceMethod<"Prints the type name.",
      "void", "printTypeB", (ins "Location":$loc),
      [{}], /*defaultImplementation=*/[{
        emitRemark(loc) << $_type << " - TestB";
      }]
    >,
    InterfaceMethod<"Prints the type name.",
      "void", "printTypeC", (ins "Location":$loc)
    >,
  ];
  let extraClassDeclaration = [{
    /// Prints the type name.
    void printTypeD(Location loc) const {
      emitRemark(loc) << *this << " - TestD";
    }
  }];
  let extraTraitClassDeclaration = [{
    /// Prints the type name.
    void printTypeE(Location loc) const {
      emitRemark(loc) << $_type << " - TestE";
    }
  }];
}

#endif // TEST_INTERFACES
