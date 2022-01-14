// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/Interfaces/SideEffectInterfaces.td"

def TEST_Dialect : Dialect {
  let name = "test";
}
class TEST_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TEST_Dialect, mnemonic, traits>;

def CustomResource : Resource<"CustomResource">;

def SideEffectOpA : TEST_Op<"side_effect_op_a"> {
  let arguments = (ins
    Arg<Variadic<AnyMemRef>, "", [MemRead]>,
    Arg<SymbolRefAttr, "", [MemRead]>:$symbol,
    Arg<FlatSymbolRefAttr, "", [MemWrite]>:$flat_symbol,
    Arg<OptionalAttr<SymbolRefAttr>, "", [MemRead]>:$optional_symbol
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc<CustomResource>]>);
}

def SideEffectOpB : TEST_Op<"side_effect_op_b",
    [MemoryEffects<[MemWrite<CustomResource>]>]>;

// CHECK: void SideEffectOpA::getEffects
// CHECK:   for (::mlir::Value value : getODSOperands(0))
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Read::get(), symbolAttr(), ::mlir::SideEffects::DefaultResource::get());
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Write::get(), flat_symbolAttr(), ::mlir::SideEffects::DefaultResource::get());
// CHECK:   if (auto symbolRef = optional_symbolAttr())
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Read::get(), symbolRef, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   for (::mlir::Value value : getODSResults(0))
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, CustomResource::get());

// CHECK: void SideEffectOpB::getEffects
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Write::get(), CustomResource::get());
