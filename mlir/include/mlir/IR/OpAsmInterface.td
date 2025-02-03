//===- OpAsmInterface.td - Asm Interfaces for opse ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces and other utilities for interacting with the
// AsmParser and AsmPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPASMINTERFACE_TD
#define MLIR_IR_OPASMINTERFACE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// OpAsmOpInterface
//===----------------------------------------------------------------------===//

def OpAsmOpInterface : OpInterface<"OpAsmOpInterface"> {
  let description = [{
    This interface provides hooks to interact with the AsmPrinter and AsmParser
    classes.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Get a special name to use when printing the results of this operation.
        The given callback is invoked with a specific result value that starts a
        result "pack", and the name to give this result pack. To signal that a
        result pack should use the default naming scheme, a None can be passed
        in instead of the name.

        For example, if you have an operation that has four results and you want
        to split these into three distinct groups you could do the following:

        ```c++
          setNameFn(getResult(0), "first_result");
          setNameFn(getResult(1), "middle_results");
          setNameFn(getResult(3), ""); // use the default numbering.
        ```

        This would print the operation as follows:

        ```mlir
          %first_result, %middle_results:2, %0 = "my.op" ...
        ```
      }],
      "void", "getAsmResultNames",
      (ins "::mlir::OpAsmSetValueNameFn":$setNameFn),
      "", "return;"
    >,
    InterfaceMethod<[{
        Get a special name to use when printing the block arguments for a region
        immediately nested under this operation.
      }],
      "void", "getAsmBlockArgumentNames",
      (ins
        "::mlir::Region&":$region,
        "::mlir::OpAsmSetValueNameFn":$setNameFn
      ),
      "", "return;"
    >,
    InterfaceMethod<[{
        Get the name to use for a given block inside a region attached to this
        operation.

        For example if this operation has multiple blocks:

        ```mlir
          some.op() ({
            ^bb0:
              ...
            ^bb1:
              ...
          })
        ```

        the method will be invoked on each of the blocks allowing the op to
        print:

        ```mlir
          some.op() ({
            ^custom_foo_name:
              ...
            ^custom_bar_name:
              ...
          })
        ```
      }],
      "void", "getAsmBlockNames",
      (ins "::mlir::OpAsmSetBlockNameFn":$setNameFn), "", ";"
    >,
    StaticInterfaceMethod<[{
      Return the default dialect used when printing/parsing operations in
      regions nested under this operation. This allows for eliding the dialect
      prefix from the operation name, for example it would be possible to omit
      the `spirv.` prefix from all operations within a SpirV module if this method
      returned `spv`. The default implementation returns an empty string which
      is ignored.
      }],
      "::llvm::StringRef", "getDefaultDialect", (ins), "", "return \"\";"
    >,
  ];
}

//===----------------------------------------------------------------------===//
// OpAsmTypeInterface
//===----------------------------------------------------------------------===//

def OpAsmTypeInterface : TypeInterface<"OpAsmTypeInterface"> {
  let description = [{
    This interface provides hooks to interact with the AsmPrinter and AsmParser
    classes.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Get a name to use when printing a value of this type.
      }],
      "void", "getAsmName",
      (ins "::mlir::OpAsmSetNameFn":$setNameFn), "", ";"
    >,
  ];
}

//===----------------------------------------------------------------------===//
// ResourceHandleParameter
//===----------------------------------------------------------------------===//

/// This parameter represents a handle to a resource that is encoded into the
/// "dialect_resources" section of the assembly format. This parameter expects a
/// C++ `handleType` that derives from `AsmDialectResourceHandleBase` and
/// implements a derived handle to the desired resource type.
class ResourceHandleParameter<string handleType, string desc = "">
    : AttrOrTypeParameter<handleType, desc> {
  let parser = "$_parser.parseResourceHandle<" # handleType # ">()";
  let printer = "$_printer.printResourceHandle($_self)";
}

#endif // MLIR_IR_OPASMINTERFACE_TD
