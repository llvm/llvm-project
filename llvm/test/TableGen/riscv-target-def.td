// RUN: llvm-tblgen -gen-riscv-target-def -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

class RISCVExtension<string name, int major, int minor, string fieldname,
                     string desc, list<SubtargetFeature> implies = [],
                     string value = "true">
    : SubtargetFeature<name, fieldname, value, desc, implies> {
  int MajorVersion = major;
  int MinorVersion = minor;
  bit Experimental = false;
}

def FeatureStdExtI
    : RISCVExtension<"i", 2, 1, "HasStdExtI",
                     "'I' (Base Integer Instruction Set)">;

def FeatureStdExtZicsr
    : RISCVExtension<"zicsr", 2, 0, "HasStdExtZicsr",
                     "'zicsr' (CSRs)">;

def FeatureStdExtZifencei
    : RISCVExtension<"zifencei", 2, 0, "HasStdExtZifencei",
                     "'Zifencei' (fence.i)">;

def Feature32Bit
    : SubtargetFeature<"32bit", "IsRV32", "true", "Implements RV32">;
def Feature64Bit
    : SubtargetFeature<"64bit", "IsRV64", "true", "Implements RV64">;

// Dummy feature that isn't an extension.
def FeatureDummy
    : SubtargetFeature<"dummy", "Dummy", "true", "Dummy">;

class RISCVProcessorModel<string n,
                          SchedMachineModel m,
                          list<SubtargetFeature> f,
                          list<SubtargetFeature> tunef = [],
                          string default_march = "">
    :  ProcessorModel<n, m, f, tunef> {
  string DefaultMarch = default_march;
}

class RISCVTuneProcessorModel<string n,
                              SchedMachineModel m,
                              list<SubtargetFeature> tunef = [],
                              list<SubtargetFeature> f = []>
    : ProcessorModel<n, m, f,tunef>;

def GENERIC_RV32 : RISCVProcessorModel<"generic-rv32",
                                       NoSchedModel,
                                       [Feature32Bit,
                                        FeatureStdExtI]>;
def GENERIC_RV64 : RISCVProcessorModel<"generic-rv64",
                                       NoSchedModel,
                                       [Feature64Bit,
                                        FeatureStdExtI]>;
def GENERIC : RISCVTuneProcessorModel<"generic", NoSchedModel>;


def ROCKET_RV32 : RISCVProcessorModel<"rocket-rv32",
                                      NoSchedModel,
                                      [Feature32Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr,
                                       FeatureDummy]>;
def ROCKET_RV64 : RISCVProcessorModel<"rocket-rv64",
                                      NoSchedModel,
                                      [Feature64Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr,
                                       FeatureDummy]>;
def ROCKET : RISCVTuneProcessorModel<"rocket",
                                     NoSchedModel>;

// CHECK: #ifndef PROC
// CHECK: #define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_UNALIGNED_ACCESS)
// CHECK: #endif

// CHECK: PROC(GENERIC_RV32, {"generic-rv32"}, {"rv32i2p1"}, 0)
// CHECK: PROC(GENERIC_RV64, {"generic-rv64"}, {"rv64i2p1"}, 0)
// CHECK: PROC(ROCKET_RV32, {"rocket-rv32"}, {"rv32i2p1_zicsr2p0_zifencei2p0"}, 0)
// CHECK: PROC(ROCKET_RV64, {"rocket-rv64"}, {"rv64i2p1_zicsr2p0_zifencei2p0"}, 0)

// CHECK: #undef PROC

// CHECK: #ifndef TUNE_PROC
// CHECK: #define TUNE_PROC(ENUM, NAME)
// CHECK: #endif

// CHECK: TUNE_PROC(GENERIC, "generic")
// CHECK: TUNE_PROC(ROCKET, "rocket")

// CHECK: #undef TUNE_PROC
