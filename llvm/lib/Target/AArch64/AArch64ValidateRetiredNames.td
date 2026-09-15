//=- AArch64ValidateRetiredNames.td - Check retired names -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check that names belonging to withdrawn architectural features and system
// registers are not reused. This file must be included after all the records
// to be checked have been instantiated.
//
//===----------------------------------------------------------------------===//

foreach Feature = !instances<SubtargetFeature>() in
  foreach RetiredName = AArch64RetiredSubtargetFeatureNames in
    assert !ne(Feature.Name, RetiredName),
        "subtarget feature name '" # RetiredName # "' has been retired";

foreach ExtensionRecord = !instances<Extension>() in {
  foreach RetiredName = AArch64RetiredSubtargetFeatureNames in {
    assert !ne(ExtensionRecord.UserVisibleName, RetiredName),
        "architecture extension name '" # RetiredName # "' has been retired";
    assert !ne(ExtensionRecord.UserVisibleAlias, RetiredName),
        "architecture extension alias '" # RetiredName # "' has been retired";
  }

  foreach RetiredName = AArch64RetiredArchFeatureNames in
    assert !eq(!find(", " # ExtensionRecord.ArchFeatureName # ", ",
                     ", " # RetiredName # ", "), -1),
        "architecture feature name '" # RetiredName # "' has been retired";
}

foreach Register = !instances<SysReg>() in
  foreach RetiredName = AArch64RetiredSystemRegisterNames in
    assert !ne(!toupper(Register.Name), !toupper(RetiredName)),
        "system register name '" # RetiredName # "' has been retired";
