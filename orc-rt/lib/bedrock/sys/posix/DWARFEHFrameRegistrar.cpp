//===- DWARFEHFrameRegistrar.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DWARF eh-frame registration support.
//
// This requires an Itanium-style unwinder providing __register_frame, not
// POSIX as such: it lives in sys/posix/ because that's the set of supported
// targets that have one.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/posix/DWARFEHFrameRegistrar.h"
#include "orc-rt-internal/support/Endian.h"
#include "orc-rt/support/Compiler.h"

#include <cstdint>

// libgcc: takes the start of a zero-terminated .eh_frame section.
extern "C" void __register_frame(const void *EHFrame) ORC_RT_WEAK_IMPORT;
extern "C" void __deregister_frame(const void *EHFrame) ORC_RT_WEAK_IMPORT;

// libunwind: takes a single FDE. (libunwind's __register_frame takes an FDE
// too, and forwards to these.)
extern "C" void __unw_add_dynamic_fde(uintptr_t FDE) ORC_RT_WEAK_IMPORT;
extern "C" void __unw_remove_dynamic_fde(uintptr_t FDE) ORC_RT_WEAK_IMPORT;

using namespace orc_rt;

namespace {

// The unwinder is chosen when the client links, not when orc-rt is built, so
// detect it at runtime. We use libunwind's own FDE API rather than calling
// __register_frame per FDE: if both unwinders are loaded, __register_frame may
// bind to libgcc's, which would misread an FDE as a section.
bool haveLibunwind() noexcept {
  return __unw_add_dynamic_fde && __unw_remove_dynamic_fde;
}

/// Call HandleFDE on the start of each FDE in EHFrame, stopping at a
/// zero-length terminator record if there is one. Returns an error if any
/// record overruns EHFrame, or if RequireTerminator is set and there's no
/// terminator.
template <typename HandleFDEFn>
Error walkEHFrameSection(span<const char> EHFrame, bool RequireTerminator,
                         HandleFDEFn &&HandleFDE) noexcept {
  const char *P = EHFrame.data();
  const char *End = P + EHFrame.size();

  auto MakeMalformedError = [&](const char *Problem) {
    return make_error<StringError>(
        std::string("Malformed .eh_frame section: ") + Problem);
  };

  while (P != End) {
    if (End - P < 4)
      return MakeMalformedError("truncated record length");
    uint64_t Length = endian_read<uint32_t>(P, endian::native);
    if (Length == 0)
      return Error::success(); // Terminator.

    size_t LengthFieldSize = 4;
    if (Length == 0xffffffff) {
      if (End - P < 12)
        return MakeMalformedError("truncated extended record length");
      Length = endian_read<uint64_t>(P + 4, endian::native);
      LengthFieldSize = 12;
    }

    // Every record starts with a 4-byte CIE id (for CIEs) or CIE pointer (for
    // FDEs).
    size_t Remaining = End - P - LengthFieldSize;
    if (Length < 4 || Length > Remaining)
      return MakeMalformedError("record overruns section");

    if (endian_read<uint32_t>(P + LengthFieldSize, endian::native) != 0)
      HandleFDE(P);

    P += LengthFieldSize + Length;
  }

  if (RequireTerminator)
    return MakeMalformedError("missing terminator");
  return Error::success();
}

} // namespace

namespace orc_rt::sys::posix {

Error DWARFEHFrameRegistrar::registerSection(
    span<const char> EHFrame) noexcept {
  if (haveLibunwind()) {
    // Validate before registering anything, so that a malformed section isn't
    // left partially registered.
    if (auto Err = walkEHFrameSection(EHFrame, false, [](const char *) {}))
      return Err;
    return walkEHFrameSection(EHFrame, false, [](const char *FDE) {
      __unw_add_dynamic_fde(reinterpret_cast<uintptr_t>(FDE));
    });
  }

  if (ORC_RT_UNLIKELY(!__register_frame))
    return make_error<StringError>("__register_frame not found");

  __register_frame(EHFrame.data());
  return Error::success();
}

Error DWARFEHFrameRegistrar::deregisterSection(
    span<const char> EHFrame) noexcept {
  if (haveLibunwind())
    return walkEHFrameSection(EHFrame, false, [](const char *FDE) {
      __unw_remove_dynamic_fde(reinterpret_cast<uintptr_t>(FDE));
    });

  if (ORC_RT_UNLIKELY(!__deregister_frame))
    return make_error<StringError>("__deregister_frame not found");

  __deregister_frame(EHFrame.data());
  return Error::success();
}

} // namespace orc_rt::sys::posix
