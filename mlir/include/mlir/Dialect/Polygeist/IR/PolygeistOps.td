#ifndef POLYGEIST_OPS
#define POLYGEIST_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/Polygeist/IR/PolygeistBase.td"

def Memref2PointerOp
    : Op<Polygeist_Dialect, "memref2pointer", [ViewLikeOpInterface, Pure]> {
  let summary = "Extract an LLVM pointer from a MemRef";

  let arguments = (ins AnyMemRef:$source);
  let results = (outs LLVM_AnyPointer:$result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def Pointer2MemrefOp
    : Op<Polygeist_Dialect, "pointer2memref", [ViewLikeOpInterface, Pure]> {
  let summary = "Upgrade a pointer to a memref";

  let arguments = (ins LLVM_AnyPointer:$source);
  let results = (outs AnyMemRef:$result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

#endif // POLYGEIST_OPS
