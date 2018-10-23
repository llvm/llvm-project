//==- DIAEnumFrameData.h --------------------------------------- -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_DIA_DIAENUMFRAMEDATA_H
#define LLVM_DEBUGINFO_PDB_DIA_DIAENUMFRAMEDATA_H

#include "DIASupport.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBFrameData.h"

namespace llvm {
namespace pdb {

class DIASession;

class DIAEnumFrameData : public IPDBEnumChildren<IPDBFrameData> {
public:
  explicit DIAEnumFrameData(const DIASession &PDBSession,
                            CComPtr<IDiaEnumFrameData> DiaEnumerator);

  uint32_t getChildCount() const override;
  ChildTypePtr getChildAtIndex(uint32_t Index) const override;
  ChildTypePtr getNext() override;
  void reset() override;

private:
  const DIASession &Session;
  CComPtr<IDiaEnumFrameData> Enumerator;
};

} // namespace pdb
} // namespace llvm

#endif
