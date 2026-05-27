//===-- RegisterTypeDetector_arm64.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERTYPEDETECTOR_ARM64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERTYPEDETECTOR_ARM64_H

#include "lldb/Target/RegisterType.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace lldb_private {

struct RegisterInfo;

/// This class manages the storage and detection of register type information.
/// The same register may have different fields on different CPUs. This class
/// abstracts out the field detection process so we can use it on live processes
/// and core files.
///
/// The way to use this class is:
/// * Make an instance somewhere that will last as long as the debug session
///   (because your final register info will point to this instance).
/// * Read hardware capabilities from a core note, binary, prctl, etc.
/// * Pass those to DetectTypes.
/// * Call UpdateRegisterInfo with your RegisterInfo to add pointers
///   to the detected types for all registers listed in this class.
///
/// This must be done in that order, and you should ensure that if multiple
/// threads will reference the information, a mutex is used to make sure only
/// one calls DetectTypes.
class Arm64RegisterTypeDetector {
public:
  /// For the registers listed in this class, detect which fields are
  /// present and build types for those. Must be called before
  /// UpdateRegisterInfos. If called more than once, fields will be redetected
  /// each time from scratch. If the target would not have this register at all,
  /// no type is produced.
  void DetectTypes(uint64_t hwcap, uint64_t hwcap2, uint64_t hwcap3);

  /// Add the type information of any registers named in this class,
  /// to the relevant RegisterInfo instances. Note that this will be done
  /// with a pointer to the instance of this class that you call this on, so
  /// the lifetime of that instance must be at least that of the register info.
  void UpdateRegisterInfo(const RegisterInfo *reg_info, uint32_t num_regs);

  /// Returns true if field detection has been run at least once.
  bool HasDetected() const { return m_has_detected; }

private:
  using DetectorFn =
      std::function<const RegisterType *(uint64_t, uint64_t, uint64_t)>;

  static const RegisterType *DetectCPSRType(uint64_t hwcap, uint64_t hwcap2,
                                            uint64_t hwcap3);
  static const RegisterType *DetectFPSRType(uint64_t hwcap, uint64_t hwcap2,
                                            uint64_t hwcap3);
  static const RegisterType *DetectFPCRType(uint64_t hwcap, uint64_t hwcap2,
                                            uint64_t hwcap3);
  static const RegisterType *DetectMTECtrlType(uint64_t hwcap, uint64_t hwcap2,
                                               uint64_t hwcap3);
  static const RegisterType *DetectSVCRType(uint64_t hwcap, uint64_t hwcap2,
                                            uint64_t hwcap3);
  static const RegisterType *DetectFPMRType(uint64_t hwcap, uint64_t hwcap2,
                                            uint64_t hwcap3);
  static const RegisterType *DetectX0Type(uint64_t hwcap, uint64_t hwcap2,
                                            uint64_t hwcap3);
  static const RegisterType *DetectV0Type(uint64_t hwcap, uint64_t hwcap2,
                                          uint64_t hwcap3);
  static const RegisterType *
  DetectGCSFeaturesType(uint64_t hwcap, uint64_t hwcap2, uint64_t hwcap3);
  static const RegisterType *DetectPOREL0Type(uint64_t hwcap, uint64_t hwcap2,
                                           uint64_t hwcap3);

  struct RegisterEntry {
    RegisterEntry(llvm::StringRef name, unsigned size, DetectorFn detector)
        : m_name(name), m_type(nullptr), m_detector(detector) {}

    llvm::StringRef m_name;
    const RegisterType *m_type;
    DetectorFn m_detector;
  } m_registers[11] = {
      RegisterEntry("cpsr", 4, DetectCPSRType),
      RegisterEntry("fpsr", 4, DetectFPSRType),
      RegisterEntry("fpcr", 4, DetectFPCRType),
      RegisterEntry("mte_ctrl", 8, DetectMTECtrlType),
      RegisterEntry("svcr", 8, DetectSVCRType),
      RegisterEntry("fpmr", 8, DetectFPMRType),
      RegisterEntry("gcs_features_enabled", 8, DetectGCSFeaturesType),
      RegisterEntry("gcs_features_locked", 8, DetectGCSFeaturesType),
      RegisterEntry("por_el0", 8, DetectPOREL0Type),
      RegisterEntry("x0", 8, DetectX0Type),
      RegisterEntry("v0", 16, DetectV0Type),
  };

  // Becomes true once field detection has been run for all registers.
  bool m_has_detected = false;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERTYPEDETECTOR_ARM64_H
