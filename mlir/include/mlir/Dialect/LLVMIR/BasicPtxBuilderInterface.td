//===- BasicPtxBuilderInterface.td - PTX builder interface -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to build PTX (Parallel Thread Execution) from NVVM Ops 
// automatically. It is used by NVVM to LLVM pass. 
//
//===----------------------------------------------------------------------===//

#ifndef BASICPTXBUILDER_OP_INTERFACE
#define BASICPTXBUILDER_OP_INTERFACE

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// Basic PTX Builder Interface
//===----------------------------------------------------------------------===//

def PtxPredicate : Optional<I1>;

def BasicPtxBuilderOpInterface : OpInterface<"BasicPtxBuilderInterface"> {
  let description = [{
    This interface is used to generate inline assembly with PTX for basic 
    operations. It's utilized in the `convert-nvvm-to-llvm pass` to lower 
    NVVM Ops that implement this interface to PTX (parallel thread execution) 
    using inline assembly Ops. Interface methods play a crucial role in this 
    lowering process.

    Here's an example of an Op with the `BasicPtxBuilderOpInterface`:    
    ```tablegen
      def NVVM_SpecialOp : NVVM_Op<"special.op",
          [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>]>,  
        Results<(outs LLVM_Type:$res)>,
        Arguments<(ins LLVM_i64ptr_any:$op1, I32:$op2)> {
        ...
        let extraClassDefinition = [{
          std::string $cppClass::getPtx() { 
            return std::string("special.op %0, %1, %2;"); 
          }
     } ];
    ```

    In the above NVVM Op example:
    ```mlir
      %0 = nvvm.special.op %1, %2 : !llvm.ptr, i32 -> i32
    ```

    The `convert-nvvm-to-llvm` pass generates the inline assembly like below. 
    The order of arguments is retained, and the read and write modifiers are 
    set based on the input and result types:
    ```mlir
      %0 = llvm.inline_asm 
                has_side_effects 
                asm_dialect = 
                att "special.op %0, %1, %2;", "=r,l,r" %arg0, %arg1 
                : (!llvm.ptr, i32) -> i32
    ```
  }];
  let cppNamespace = "::mlir::NVVM";
  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          Optional function for setting a predicate, which 
          always returns a `PtxPredicate` value of type i1. If no predicate is 
          provided, the instruction is unguarded; otherwise, it's guarded by the 
          predicate value. The `PtxPredicate` value must always be the last argument. 
          The provided PTX code by `getPtx` should not include the predicate usage.
          The interface automatically handles predicate usage in the generated
          PTX code when necessary.
        }],
        /*retType=*/"std::optional<::mlir::Value>",
        /*methodName=*/"getPredicate",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return {};"
      >,
    InterfaceMethod<
        /*desc=*/[{ Returns PTX assembly with operand number. }],
        /*retType=*/"std::string",
        /*methodName=*/"getPtx"
      >,
    InterfaceMethod<
        /*desc=*/[{
          This function indicates whether the operation is supported by LLVM 
          intrinsics. It's particularly useful for operations that have 
          specific cases with LLVM intrinsic support.
        }],
        /*retType=*/"bool",
        /*methodName=*/"hasIntrinsic",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return false;"
      >,
    InterfaceMethod<
        /*desc=*/[{Return whether the operation has memory side effects.}],
        /*retType=*/"bool",
        /*methodName=*/"hasSideEffect",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return true;"
      >,
    
    InterfaceMethod<
        /*desc=*/[{Helper function to generate i32 constant value.}],
        /*retType=*/"::mlir::Value",
        /*methodName=*/"makeConstantI32",
        /*args=*/(ins "::mlir::RewriterBase &":$rewriter, "int" : $val),
        /*methodBody=*/"",
        /*defaultImpl=*/ [{
            mlir::Operation* op = $_op;
            return LLVM::ConstantOp::create(rewriter,
              op->getLoc(), rewriter.getIntegerType(32), val);
        }]
     >,
     InterfaceMethod<
         /*desc=*/[{ 
            This function supplies the necessary arguments for passing PTX code,
            following this order:
             1) Adds results 
             2) Adds operands 
             3) Adds attributes             
          }],
         /*retType=*/"void",
         /*methodName=*/"getAsmValues",
         /*args=*/(ins "::mlir::RewriterBase &":$rewriter, 
         "llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>&" : $asmValues),
         /*methodBody=*/"",
         /*defaultImpl=*/ [{         
           mlir::Operation* op = $_op;
           
           // Step 1. Add results
           for (auto val : op->getResults()) 
            asmValues.push_back({val, mlir::NVVM::PTXRegisterMod::Write});

           // Step 2. Add operands
           for (auto val : op->getOperands()) 
            asmValues.push_back({val, mlir::NVVM::PTXRegisterMod::Read});
           
           // Step 3. Add attributes
           for (auto attr : op->getAttrs()) {
            if (auto intAttr = dyn_cast<mlir::IntegerAttr>(attr.getValue())) {
             ::mlir::Value val = makeConstantI32(rewriter, intAttr.getInt());
             asmValues.push_back({val, mlir::NVVM::PTXRegisterMod::Read});
             }
           }
         }]
       >
  ];
}

#endif // BASICPTXBUILDER_OP_INTERFACE