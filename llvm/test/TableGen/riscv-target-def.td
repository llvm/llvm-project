// RUN: llvm-tblgen -gen-riscv-target-def -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

class RISCVExtension<string name, int major, int minor, string desc,
                     list<SubtargetFeature> implies = [],
                     string fieldname = !subst("Feature", "Has", NAME),
                     string value = "true">
    : SubtargetFeature<name, fieldname, value, desc, implies> {
  int MajorVersion = major;
  int MinorVersion = minor;
  bit Experimental = false;
}

class RISCVExperimentalExtension<string name, int major, int minor, string desc,
                                 list<RISCVExtension> implies = [],
                                 string fieldname = !subst("Feature", "Has", NAME),
                                 string value = "true">
    : RISCVExtension<"experimental-"#name, major, minor, desc, implies,
                     fieldname, value> {
  let Experimental = true;
}

def FeatureStdExtI
    : RISCVExtension<"i", 2, 1,
                     "'I' (Base Integer Instruction Set)">;

def FeatureStdExtZicsr
    : RISCVExtension<"zicsr", 2, 0,
                     "'zicsr' (CSRs)">;

def FeatureStdExtZifencei
    : RISCVExtension<"zifencei", 2, 0,
                     "'Zifencei' (fence.i)">;

def FeatureStdExtF
    : RISCVExtension<"f", 2, 2,
                     "'F' (Single-Precision Floating-Point)",
                     [FeatureStdExtZicsr]>;

def FeatureStdExtZidummy
    : RISCVExperimentalExtension<"zidummy", 0, 1,
                                 "Dummy">;

def Feature32Bit
    : SubtargetFeature<"32bit", "IsRV32", "true", "Implements RV32">;
def Feature64Bit
    : SubtargetFeature<"64bit", "IsRV64", "true", "Implements RV64">;

// Dummy feature that isn't an extension.
def FeatureDummy
    : SubtargetFeature<"dummy", "Dummy", "true", "Dummy">;

class RISCVProfile<string name, list<SubtargetFeature> features>
    : SubtargetFeature<name, "Is" # NAME, "true",
                       "RISC-V " # name # " profile", features> {
  bit Experimental = false;
}
class RISCVExperimentalProfile<string name, list<SubtargetFeature> features>
    : RISCVProfile<"experimental-"#name, features> {
  let Experimental = true;
}

def RVI20U32 : RISCVProfile<"rvi20u32", [Feature32Bit, FeatureStdExtI]>;
def RVI20U64 : RISCVProfile<"rvi20u64", [Feature64Bit, FeatureStdExtI]>;
def ProfileDummy : RISCVProfile<"dummy", [Feature64Bit, FeatureStdExtI,
                                          FeatureStdExtF, FeatureStdExtZidummy]>;
def RVI99U64 : RISCVExperimentalProfile<"rvi99u64", [Feature64Bit, FeatureStdExtI]>;

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
                                       FeatureStdExtZidummy,
                                       FeatureDummy]>;
def ROCKET_RV64 : RISCVProcessorModel<"rocket-rv64",
                                      NoSchedModel,
                                      [Feature64Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr,
                                       FeatureStdExtZidummy,
                                       FeatureDummy]>;
def ROCKET : RISCVTuneProcessorModel<"rocket",
                                     NoSchedModel>;

// CHECK:      #ifdef GET_SUPPORTED_EXTENSIONS
// CHECK-NEXT: #undef GET_SUPPORTED_EXTENSIONS

// CHECK:      static const RISCVSupportedExtension SupportedExtensions[] = {
// CHECK-NEXT:     {"f", {2, 2}},
// CHECK-NEXT:     {"i", {2, 1}},
// CHECK-NEXT:     {"zicsr", {2, 0}},
// CHECK-NEXT:     {"zifencei", {2, 0}},
// CHECK-NEXT: };

// CHECK:      static const RISCVSupportedExtension SupportedExperimentalExtensions[] = {
// CHECK-NEXT:     {"zidummy", {0, 1}},
// CHECK-NEXT: };

// CHECK:      #endif // GET_SUPPORTED_EXTENSIONS

// CHECK:      #ifdef GET_IMPLIED_EXTENSIONS
// CHECK-NEXT: #undef GET_IMPLIED_EXTENSIONS

// CHECK:      static constexpr ImpliedExtsEntry ImpliedExts[] = {
// CHECK-NEXT:       { {"f"}, "zicsr"},
// CHECK-NEXT: };

// CHECK:      #endif // GET_IMPLIED_EXTENSIONS

// CHECK:      #ifdef GET_SUPPORTED_PROFILES
// CHECK-NEXT: #undef GET_SUPPORTED_PROFILES

// CHECK:      static constexpr RISCVProfile SupportedProfiles[] = {
// CHECK-NEXT:     {"dummy","rv64i2p1_f2p2_zidummy0p1"},
// CHECK-NEXT:     {"rvi20u32","rv32i2p1"},
// CHECK-NEXT:     {"rvi20u64","rv64i2p1"},
// CHECK-NEXT: };

// CHECK:      static constexpr RISCVProfile SupportedExperimentalProfiles[] = {
// CHECK-NEXT:     {"rvi99u64","rv64i2p1"},
// CHECK-NEXT: };

// CHECK:      #endif // GET_SUPPORTED_PROFILES

// CHECK:      #ifndef PROC
// CHECK-NEXT: #define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_UNALIGNED_ACCESS)
// CHECK-NEXT: #endif

// CHECK:      PROC(GENERIC_RV32, {"generic-rv32"}, {"rv32i2p1"}, 0)
// CHECK-NEXT: PROC(GENERIC_RV64, {"generic-rv64"}, {"rv64i2p1"}, 0)
// CHECK-NEXT: PROC(ROCKET_RV32, {"rocket-rv32"}, {"rv32i2p1_zicsr2p0_zidummy0p1_zifencei2p0"}, 0)
// CHECK-NEXT: PROC(ROCKET_RV64, {"rocket-rv64"}, {"rv64i2p1_zicsr2p0_zidummy0p1_zifencei2p0"}, 0)

// CHECK: #undef PROC

// CHECK:      #ifndef TUNE_PROC
// CHECK-NEXT: #define TUNE_PROC(ENUM, NAME)
// CHECK-NEXT: #endif

// CHECK: TUNE_PROC(GENERIC, "generic")
// CHECK-NEXT: TUNE_PROC(ROCKET, "rocket")

// CHECK: #undef TUNE_PROC
