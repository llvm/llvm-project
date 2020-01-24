// This checks that we can consume LLVM's Intrinsic definitions from TableGen
// files and produce ODS. Unlike MLIR, LLVM's main Intrinsics.td file that
// contains the definition of the Intrinsic class also includes files for
// platform-specific intrinsics, so we need to give it to TableGen instead of
// writing a local test source. We filter out platform-specific intrinsic
// includes from the main file to avoid unnecessary dependencies and decrease
// the test cost. The command-line flags further ensure a specific intrinsic is
// processed and we only check the ouptut below.
// We also verify emission of type specialization for overloadable intrinsics.
//
// RUN: cat %S/../../../llvm/include/llvm/IR/Intrinsics.td \
// RUN: | grep -v "llvm/IR/Intrinsics" \
// RUN: | mlir-tblgen -gen-llvmir-intrinsics -I %S/../../../llvm/include/ --llvmir-intrinsics-filter=ptrmask \
// RUN: | FileCheck %s

// CHECK-LABEL: def LLVM_ptrmask
// CHECK: LLVM_IntrOp<"ptrmask
// CHECK: Arguments<(ins
// CHECK: Results<(outs
// CHECK: llvm::Function *fn = llvm::Intrinsic::getDeclaration(
// CHECK:        module, llvm::Intrinsic::ptrmask, {
// CHECK:        opInst.getResult(0).getType().cast<LLVM::LLVMType>().getUnderlyingType(),
// CHECK:        opInst.getOperand(0).getType().cast<LLVM::LLVMType>().getUnderlyingType(),
// CHECK:        opInst.getOperand(1).getType().cast<LLVM::LLVMType>().getUnderlyingType(),
// CHECK: });
// CHECK:        lookupValues(opInst.getOperands());

//---------------------------------------------------------------------------//

// This checks that the ODS we produce can be consumed by MLIR tablegen. We only
// make sure the entire process does not fail and produces some C++. The shape
// of this C++ code is tested by ODS tests.

// RUN: cat %S/../../../llvm/include/llvm/IR/Intrinsics.td \
// RUN: | grep -v "llvm/IR/Intrinsics" \
// RUN: | mlir-tblgen -gen-llvmir-intrinsics -I %S/../../../llvm/include/ --llvmir-intrinsics-filter=vastart \
// RUN: | mlir-tblgen -gen-op-decls -I %S/../../include \
// RUN: | FileCheck --check-prefix=ODS %s

// ODS-LABEL: class vastart

// RUN: cat %S/../../../llvm/include/llvm/IR/Intrinsics.td \
// RUN: | grep -v "llvm/IR/Intrinsics" \
// RUN: | mlir-tblgen -gen-llvmir-intrinsics -I %S/../../../llvm/include/ --llvmir-intrinsics-filter=ptrmask -dialect-opclass-base My_OpBase \
// RUN: | FileCheck %s --check-prefix=DIALECT-OPBASE

// DIALECT-OPBASE-LABEL: def LLVM_ptrmask
// DIALECT-OPBASE: My_OpBase<"ptrmask
