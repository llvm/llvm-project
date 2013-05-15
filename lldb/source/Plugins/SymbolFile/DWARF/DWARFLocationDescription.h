//===-- DWARFLocationDescription.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFLocationDescription_h_
#define SymbolFileDWARF_DWARFLocationDescription_h_

#include "SymbolFileDWARF.h"

int
print_dwarf_expression (lldb_private::Stream &s,
                        const lldb_private::DataExtractor& data,
                        int address_size,
                        int dwarf_ref_size,
                        bool location_expression);



#endif  // SymbolFileDWARF_DWARFLocationDescription_h_
