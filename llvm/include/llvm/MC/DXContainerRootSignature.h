//===- llvm/MC/DXContainerRootSignature.h - RootSignature -*- C++ -*- ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>
#include <variant>

namespace llvm {

class raw_ostream;
namespace mcdxbc {

struct RootParameterHeader : public dxbc::RootParameterHeader {

  size_t Location;

  RootParameterHeader() = default;

  RootParameterHeader(dxbc::RootParameterHeader H, size_t L)
      : dxbc::RootParameterHeader(H), Location(L) {}
};

using RootDescriptor = std::variant<dxbc::RST0::v0::RootDescriptor,
                                    dxbc::RST0::v1::RootDescriptor>;
using ParametersView =
    std::variant<dxbc::RootConstants, dxbc::RST0::v0::RootDescriptor,
                 dxbc::RST0::v1::RootDescriptor>;
struct RootParameter {
  SmallVector<RootParameterHeader> Headers;

  SmallVector<dxbc::RootConstants> Constants;
  SmallVector<RootDescriptor> Descriptors;

  void addHeader(dxbc::RootParameterHeader H, size_t L) {
    Headers.push_back(RootParameterHeader(H, L));
  }

  void addParameter(dxbc::RootParameterHeader H, dxbc::RootConstants C) {
    addHeader(H, Constants.size());
    Constants.push_back(C);
  }

  void addParameter(dxbc::RootParameterHeader H,
                    dxbc::RST0::v0::RootDescriptor D) {
    addHeader(H, Descriptors.size());
    Descriptors.push_back(D);
  }

  void addParameter(dxbc::RootParameterHeader H,
                    dxbc::RST0::v1::RootDescriptor D) {
    addHeader(H, Descriptors.size());
    Descriptors.push_back(D);
  }

  ParametersView get(const RootParameterHeader &H) const {
    switch (H.ParameterType) {
    case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit):
      return Constants[H.Location];
    case llvm::to_underlying(dxbc::RootParameterType::CBV):
    case llvm::to_underlying(dxbc::RootParameterType::SRV):
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
      RootDescriptor VersionedParam = Descriptors[H.Location];
      if (std::holds_alternative<dxbc::RST0::v0::RootDescriptor>(
              VersionedParam))
        return std::get<dxbc::RST0::v0::RootDescriptor>(VersionedParam);
      return std::get<dxbc::RST0::v1::RootDescriptor>(VersionedParam);
    }

    llvm_unreachable("Unimplemented parameter type");
  }

  struct iterator {
    const RootParameter &Parameters;
    SmallVector<RootParameterHeader>::const_iterator Current;

    // Changed parameter type to match member variable (removed const)
    iterator(const RootParameter &P,
             SmallVector<RootParameterHeader>::const_iterator C)
        : Parameters(P), Current(C) {}
    iterator(const iterator &) = default;

    ParametersView operator*() {
      ParametersView Val;
      switch (Current->ParameterType) {
      case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit):
        Val = Parameters.Constants[Current->Location];
        break;

      case llvm::to_underlying(dxbc::RootParameterType::CBV):
      case llvm::to_underlying(dxbc::RootParameterType::SRV):
      case llvm::to_underlying(dxbc::RootParameterType::UAV):
        RootDescriptor VersionedParam =
            Parameters.Descriptors[Current->Location];
        if (std::holds_alternative<dxbc::RST0::v0::RootDescriptor>(
                VersionedParam))
          Val = std::get<dxbc::RST0::v0::RootDescriptor>(VersionedParam);
        else
          Val = std::get<dxbc::RST0::v1::RootDescriptor>(VersionedParam);
        break;
      }
      return Val;
    }

    iterator operator++() {
      Current++;
      return *this;
    }

    iterator operator++(int) {
      iterator Tmp = *this;
      ++*this;
      return Tmp;
    }

    iterator operator--() {
      Current--;
      return *this;
    }

    iterator operator--(int) {
      iterator Tmp = *this;
      --*this;
      return Tmp;
    }

    bool operator==(const iterator I) { return I.Current == Current; }
    bool operator!=(const iterator I) { return !(*this == I); }
  };

  iterator begin() const { return iterator(*this, Headers.begin()); }

  iterator end() const { return iterator(*this, Headers.end()); }

  size_t size() const { return Headers.size(); }

  bool isEmpty() const { return Headers.empty(); }

  llvm::iterator_range<RootParameter::iterator> getAll() const {
    return llvm::make_range(begin(), end());
  }
};
struct RootSignatureDesc {

  uint32_t Version = 2U;
  uint32_t Flags = 0U;
  uint32_t RootParameterOffset = 0U;
  uint32_t StaticSamplersOffset = 0u;
  uint32_t NumStaticSamplers = 0u;
  mcdxbc::RootParameter Parameters;

  void write(raw_ostream &OS) const;

  size_t getSize() const;
};
} // namespace mcdxbc
} // namespace llvm
