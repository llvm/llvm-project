// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/Interfaces/SideEffectInterfaces.td"

def TEST_Dialect : Dialect {
  let name = "test";
}
class TEST_Op<string mnemonic, list<Trait> traits = []> :
    Op<TEST_Dialect, mnemonic, traits>;

def CustomResource : Resource<"CustomResource">;

def SideEffectOpA : TEST_Op<"side_effect_op_a"> {
  let arguments = (ins
    Arg<Variadic<AnyMemRef>, "", [MemRead]>,
    Arg<AnyMemRef, "", [MemWriteAt<1, FullEffect>]>,
    Arg<SymbolRefAttr, "", [MemRead]>:$symbol,
    Arg<FlatSymbolRefAttr, "", [MemWrite]>:$flat_symbol,
    Arg<OptionalAttr<SymbolRefAttr>, "", [MemRead]>:$optional_symbol
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc<CustomResource, 0>]>);
}

def SideEffectOpB : TEST_Op<"side_effect_op_b",
    [MemoryEffects<[MemWrite<CustomResource, 0>]>]>;

// CHECK: void SideEffectOpA::getEffects
// CHECK:  {
// CHECK:    auto valueRange = getODSOperandIndexAndLength(0);
// CHECK:    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
// CHECK:      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::SideEffects::DefaultResource::get());
// CHECK:    }
// CHECK:  }
// CHECK:  {
// CHECK:    auto valueRange = getODSOperandIndexAndLength(1);
// CHECK:    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
// CHECK:      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 1, true, ::mlir::SideEffects::DefaultResource::get());
// CHECK:    }
// CHECK:  }
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Read::get(), getSymbolAttr(), 0, false, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Write::get(), getFlatSymbolAttr(), 0, false, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   if (auto symbolRef = getOptionalSymbolAttr())
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Read::get(), symbolRef, 0, false, ::mlir::SideEffects::DefaultResource::get());
// CHECK:  {
// CHECK:    auto valueRange = getODSResultIndexAndLength(0);
// CHECK:    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
// CHECK:      effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), getOperation()->getOpResult(idx), 0, false, CustomResource::get());
// CHECK:    }
// CHECK:  }

// CHECK: void SideEffectOpB::getEffects
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Write::get(), 0, false, CustomResource::get());
