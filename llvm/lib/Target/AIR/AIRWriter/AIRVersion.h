//===- AIRVersion.h - macOS-version-driven AIR version derivation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Single source of truth for the per-target-macOS AIR/metallib version
// constants. All values below were EMPIRICALLY VERIFIED against Apple's own
// compiler via `xcrun metal -mmacosx-version-min=N`:
//
//   macOS | triple                          | air.version | MTLB VERS air_minor
//   ------+---------------------------------+-------------+--------------------
//    13   | air64_v25-apple-macosx13.0.0    | (2,5,0)     | 5
//    14   | air64_v26-apple-macosx14.0.0    | (2,6,0)     | 6
//    15   | air64_v27-apple-macosx15.0.0    | (2,7,0)     | 7
//    16   | air64_v28-apple-macosx26.0.0    | (2,8,0)     | 8
//
// Derivation patterns:
//   * subarch `_vNN`       : NN = OSmajor + 12   (13->25, 14->26, ...)
//   * air.version minor    :      OSmajor - 8    (13->5, 14->6, ...)
//   * air.version major    : always 2
//   * triple OS component  : OSmajor, EXCEPT 16 -> 26 (Apple's renumber);
//                            for 13/14/15 it stays macosxNN.0.0.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_AIRVERSION_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_AIRVERSION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cctype>
#include <string>

namespace llvm {
namespace metal {

/// Derives all per-target AIR/metallib version constants from the target
/// macOS major version. See file header for the empirical table.
struct AIRVersion {
  unsigned OSMajor;       // 13,14,15,16
  unsigned AIRMinor;      // OSMajor - 8 (air.version / VERS minor)
  unsigned SubarchVer;    // OSMajor + 12 (the `_vNN` subarch)
  unsigned TripleOSMajor; // OSMajor, except 16 -> 26 (Apple renumber)
  unsigned MSLMajor;      // AIR Shading Language major: <=15 -> 3, >=16 -> 4
  unsigned MSLMinor;      // 13->0, 14->1, 15->2, 16->0  (from `xcrun metal`)
  unsigned ContainerByte; // MTLB header byte 8: 13->7,14->7,15->8,16->9

  /// air.version major is always 2 across all supported macOS versions.
  static constexpr unsigned AIRMajor = 2;

  static AIRVersion fromOSMajor(unsigned OS) {
    unsigned TripleOS = (OS >= 16) ? OS + 10 : OS;
    unsigned MSLMaj = (OS >= 16) ? 4 : 3;
    unsigned MSLMin = (OS >= 16) ? 0 : (OS - 13);
    // MTLB container format byte (header byte 8), from `xcrun metal`:
    // 13->7, 14->7, 15->8, 16->9.
    unsigned CByte = (OS <= 14) ? 7 : (OS == 15 ? 8 : 9);
    return {OS, OS - 8, OS + 12, TripleOS, MSLMaj, MSLMin, CByte};
  }

  /// Parse the macOS major from a triple string such as
  /// "air64_v28-apple-macosx26.0.0". The number after "macosx" maps back:
  /// 26 -> 16 (Apple renumber), otherwise NN -> NN (13/14/15). Defaults to
  /// OSMajor=16 when no "macosx" component is found, preserving current
  /// shipping behavior (macOS 16 / 26-era).
  static AIRVersion fromTriple(llvm::StringRef Triple) {
    constexpr llvm::StringRef Marker = "macosx";
    size_t Pos = Triple.find(Marker);
    if (Pos == llvm::StringRef::npos)
      return fromOSMajor(16);

    llvm::StringRef Rest = Triple.substr(Pos + Marker.size());
    // Scan leading digits of the OS major number.
    unsigned Num = 0;
    size_t I = 0;
    for (; I < Rest.size() && std::isdigit((unsigned char)Rest[I]); ++I)
      Num = Num * 10 + (unsigned)(Rest[I] - '0');
    if (I == 0)
      return fromOSMajor(16); // no digits -> fallback

    unsigned OS = (Num >= 26) ? Num - 10 : Num;
    return fromOSMajor(OS);
  }

  /// Reconstruct the canonical AIR triple, e.g. "air64_v28-apple-macosx26.0.0".
  std::string tripleString() const {
    return ("air64_v" + llvm::Twine(SubarchVer) + "-apple-macosx" +
            llvm::Twine(TripleOSMajor) + ".0.0")
        .str();
  }
};

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRWRITER_AIRVERSION_H
