//===-- DWARFDataExtractor.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDataExtractor.h"
#include "llvm/ADT/ArrayRef.h"

namespace lldb_private {

llvm::DWARFDataExtractor DWARFDataExtractor::GetAsLLVMDWARF() const {
  return llvm::DWARFDataExtractor(llvm::ArrayRef(GetDataStart(), GetByteSize()),
                                  GetByteOrder() == lldb::eByteOrderLittle,
                                  GetAddressByteSize());
}
llvm::DataExtractor DWARFDataExtractor::GetAsLLVM() const {
  return llvm::DataExtractor(llvm::ArrayRef(GetDataStart(), GetByteSize()),
                             GetByteOrder() == lldb::eByteOrderLittle,
                             GetAddressByteSize());
}
} // namespace lldb_private
