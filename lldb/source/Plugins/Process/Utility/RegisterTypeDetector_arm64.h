//===-- RegisterTypeDetector_arm64.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERTYPEDETECTOR_ARM64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERTYPEDETECTOR_ARM64_H

#include "lldb/Utility/RegisterInfo.h"
#include "lldb/Utility/RegisterType.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace lldb_private {

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
  // A detector function inspects the hwcaps and builds a type for that
  // register. All types should be made using MakeType, and a raw pointer to
  // the top level type must be returned.
  using DetectorFn = const RegisterType *(
      Arm64RegisterTypeDetector::*)(uint64_t, uint64_t, uint64_t);

  const RegisterType *DetectCPSRType(uint64_t hwcap, uint64_t hwcap2,
                                     uint64_t hwcap3);
  const RegisterType *DetectFPSRType(uint64_t hwcap, uint64_t hwcap2,
                                     uint64_t hwcap3);
  const RegisterType *DetectFPCRType(uint64_t hwcap, uint64_t hwcap2,
                                     uint64_t hwcap3);
  const RegisterType *DetectMTECtrlType(uint64_t hwcap, uint64_t hwcap2,
                                        uint64_t hwcap3);
  const RegisterType *DetectSVCRType(uint64_t hwcap, uint64_t hwcap2,
                                     uint64_t hwcap3);
  const RegisterType *DetectFPMRType(uint64_t hwcap, uint64_t hwcap2,
                                     uint64_t hwcap3);
  const RegisterType *DetectGCSFeaturesType(uint64_t hwcap, uint64_t hwcap2,
                                            uint64_t hwcap3);
  const RegisterType *DetectPOREL0Type(uint64_t hwcap, uint64_t hwcap2,
                                       uint64_t hwcap3);

  struct RegisterEntry {
    RegisterEntry(const std::vector<llvm::StringRef> &names,
                  DetectorFn detector)
        : m_names(names), m_type(nullptr), m_detector(detector) {}

    std::vector<llvm::StringRef> m_names;
    // A raw pointer to the top level type. This pointer's lifetime is managed
    // by a unique pointer of the same value in m_detected_types.
    const RegisterType *m_type;
    DetectorFn m_detector;
  } m_registers[8] = {
      RegisterEntry({"cpsr"}, &Arm64RegisterTypeDetector::DetectCPSRType),
      RegisterEntry({"fpsr"}, &Arm64RegisterTypeDetector::DetectFPSRType),
      RegisterEntry({"fpcr"}, &Arm64RegisterTypeDetector::DetectFPCRType),
      RegisterEntry({"mte_ctrl"},
                    &Arm64RegisterTypeDetector::DetectMTECtrlType),
      RegisterEntry({"svcr"}, &Arm64RegisterTypeDetector::DetectSVCRType),
      RegisterEntry({"fpmr"}, &Arm64RegisterTypeDetector::DetectFPMRType),
      RegisterEntry({"gcs_features_enabled", "gcs_features_locked"},
                    &Arm64RegisterTypeDetector::DetectGCSFeaturesType),
      RegisterEntry({"por_el0"}, &Arm64RegisterTypeDetector::DetectPOREL0Type),
  };

  // Becomes true once field detection has been run for all registers.
  bool m_has_detected = false;

  template <typename T, typename... Args> const T *MakeType(Args &&...args) {
    static_assert(std::is_base_of_v<RegisterType, T>);

    auto type = std::make_unique<T>(std::forward<Args>(args)...);
    const T *type_ptr = type.get();
    m_detected_types.detected_types.push_back(std::move(type));
    return type_ptr;
  }

  // This stores all the types created. There may be > 1 type per register,
  // as a register may nest types (enums for fields for example).
  // We do not use a vector of RegisterType, because the address of the types
  // must remain the same as new types are created.
  // Code other than MakeType should not use this vector directly, hence the
  // class wrapper to enforce that.
  class DetectedTypesHolder {
    std::vector<std::unique_ptr<RegisterType>> detected_types;

    template <typename T, typename... Args>
    friend const T *Arm64RegisterTypeDetector::MakeType(Args &&...args);
  } m_detected_types;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERTYPEDETECTOR_ARM64_H
