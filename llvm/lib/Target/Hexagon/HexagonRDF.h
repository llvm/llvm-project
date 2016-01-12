//===--- HexagonRDF.h -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_RDF_H
#define HEXAGON_RDF_H
#include "RDFGraph.h"

namespace llvm {
  class TargetRegisterInfo;
}

namespace rdf {
  struct HexagonRegisterAliasInfo : public RegisterAliasInfo {
    HexagonRegisterAliasInfo(const TargetRegisterInfo &TRI)
      : RegisterAliasInfo(TRI) {}
    bool covers(RegisterRef RA, RegisterRef RR) const override;
    bool covers(const RegisterSet &RRs, RegisterRef RR) const override;
  };
}

#endif

