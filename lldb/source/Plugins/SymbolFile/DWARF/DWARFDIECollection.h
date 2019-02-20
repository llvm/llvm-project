//===-- DWARFDIECollection.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDIECollection_h_
#define SymbolFileDWARF_DWARFDIECollection_h_

#include "DWARFDIE.h"
#include <vector>

class DWARFDIECollection {
public:
  DWARFDIECollection() : m_dies() {}
  ~DWARFDIECollection() {}

  void Append(const DWARFDIE &die);

  void Dump(lldb_private::Stream *s, const char *title) const;

  DWARFDIE
  GetDIEAtIndex(uint32_t idx) const;

  size_t Size() const;

protected:
  typedef std::vector<DWARFDIE> collection;
  typedef collection::iterator iterator;
  typedef collection::const_iterator const_iterator;

  collection m_dies; // Ordered list of die offsets
};

#endif // SymbolFileDWARF_DWARFDIECollection_h_
