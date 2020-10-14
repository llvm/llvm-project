// REQUIRES: asserts
// REQUIRES: aarch64-registered-target
// RUN: llvm-tblgen -gen-instr-info %s -I%p/../../include -I%p/../../lib/Target/AArch64 -o %t -debug-only=subtarget-emitter 2>&1 | FileCheck %s 

// Check that we've defined scheduling classes for FMOVv2f32_ns and FMOVv2f64 for Model0
// CHECK: InstRW: New SC [[SC:[0-9]+]]:FMOVv2f32_ns on Model0
// CHECK: InstRW: New SC [[SC2:[0-9]+]]:FMOVv2f64_ns on Model0

// Generic transition for WriteV should be defined for Model0 as well as for
// all instructions without explicitly defined scheduling classes.
// CHECK: Adding transition from WriteV({{[0-9]+}}) to Model0WriteV_4cyc({{[0-9]+}}) on processor indices
// CHECK: Adding transition from WriteV({{[0-9]+}}) to Model0WriteV_2cyc({{[0-9]+}}) on processor indices

// Transition from FMOVv2f64_ns should still be added for Model0,
// even though we've defined custom scheduling class.
// CHECK: Adding transition from FMOVv2f64_ns([[SC2]]) to Model0WriteV_4cyc({{[0-9]+}}) on processor indices
// CHECK-NEXT: Adding transition from FMOVv2f64_ns([[SC2]]) to Model0WriteV_2cyc({{[0-9]+}}) on processor indices

// Transition from FMOVv2f32_ns should not be added for Model0,
// because custom sched class for it is defined and it's not variant.
// CHECK-NOT: Adding transition from FMOVv2f32_ns([[SC]])

include "AArch64.td"

def Model0 : SchedMachineModel {
  let CompleteModel = 0;
}

def Model0UnitV    : ProcResource<1> { let BufferSize = 0; }

let SchedModel = Model0 in {

def Model0WriteV_4cyc : SchedWriteRes<[Model0UnitV]> { let Latency = 4; }
def Model0WriteV_2cyc : SchedWriteRes<[Model0UnitV]> { let Latency = 2; }
def Model0WriteV_1cyc : SchedWriteRes<[Model0UnitV]> { let Latency = 1; }

def Model0QFormPred : MCSchedPredicate<CheckQForm>;
def Model0WriteV : SchedWriteVariant<[
       SchedVar<Model0QFormPred, [Model0WriteV_4cyc]>,
       SchedVar<NoSchedPred, [Model0WriteV_2cyc]>]>;

def : SchedAlias<WriteV, Model0WriteV>;

def : InstRW<[Model0WriteV_1cyc], (instrs FMOVv2f32_ns)>;
def : InstRW<[WriteV], (instrs FMOVv2f64_ns)>;
}

def : ProcessorModel<"foo-0-model", Model0, []>;
