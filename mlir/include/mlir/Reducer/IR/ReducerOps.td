#ifndef REDUCER_OPS
#define REDUCER_OPS

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

def Reducer_Dialect : Dialect {
  let name = "reducer";
  let cppNamespace = "::mlir::reducer";
}

// Base class for Reducer dialect ops.
class Reducer_Op<string mnemonic, list<Trait> traits = []>
    : Op<Reducer_Dialect, mnemonic, traits>;

def BarrierOp : Reducer_Op<"barrier", []> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    ($inputs^ `:` type($inputs))? `to` type($outputs) attr-dict
  }];
}

#endif // REDUCER_OPS