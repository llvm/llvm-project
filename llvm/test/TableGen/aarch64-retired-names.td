// RUN: not llvm-tblgen -print-records -DREUSE_RETIRED_NAMES %s \
// RUN:   -I %p/../../lib/Target/AArch64 -I %p/../../include 2>&1 | \
// RUN:   FileCheck %s
include "AArch64.td"

#ifdef REUSE_RETIRED_NAMES
foreach RetiredName = AArch64RetiredSubtargetFeatureNames in
  def : SubtargetFeature<RetiredName, "ReusedFeature", "true",
                         "Invalid test feature">;

foreach RetiredName = AArch64RetiredSubtargetFeatureNames in {
  let UserVisibleName = RetiredName in
  def : Extension<"test-name-" # RetiredName, "TestExtensionName",
                  "FEAT_TEST", "Invalid test extension">;

  let UserVisibleAlias = RetiredName in
  def : Extension<"test-alias-" # RetiredName, "TestExtensionAlias",
                  "FEAT_TEST", "Invalid test extension">;
}

foreach RetiredName = AArch64RetiredArchFeatureNames in
  def : Extension<"test-" # RetiredName, "TestArchFeature",
                  "FEAT_TEST, " # RetiredName, "Invalid test extension">;

foreach RetiredName = AArch64RetiredSystemRegisterNames in
  def : ROSysReg<!tolower(RetiredName), 0b11, 0b000, 0b0000, 0b0000, 0b000>;
#endif

// Re-run the checks after instantiating the deliberately invalid record.
include "AArch64ValidateRetiredNames.td"

// CHECK-DAG: assertion failed: subtarget feature name 'mpamv2' has been retired
// CHECK-DAG: assertion failed: subtarget feature name 'tme' has been retired
// CHECK-DAG: assertion failed: architecture extension name 'mpamv2' has been retired
// CHECK-DAG: assertion failed: architecture extension name 'tme' has been retired
// CHECK-DAG: assertion failed: architecture extension alias 'mpamv2' has been retired
// CHECK-DAG: assertion failed: architecture extension alias 'tme' has been retired
// CHECK-DAG: assertion failed: architecture feature name 'FEAT_MPAMv2' has been retired
// CHECK-DAG: assertion failed: architecture feature name 'FEAT_MPAMv2_VID' has been retired
// CHECK-DAG: assertion failed: architecture feature name 'FEAT_TME' has been retired
// CHECK-DAG: assertion failed: system register name 'MPAMVIDCR_EL2' has been retired
// CHECK-DAG: assertion failed: system register name 'MPAMVIDSR_EL2' has been retired
// CHECK-DAG: assertion failed: system register name 'MPAMVIDSR_EL3' has been retired
