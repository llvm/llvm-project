// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_TABLEGEN_CAPI_H_
#define MLIR_TABLEGEN_CAPI_H_

#include "mlir/Support/LLVM.h"
#include "llvm/TableGen/Record.h"

namespace mlir {
namespace tblgen {

bool emitPassCAPIImpl(const llvm::RecordKeeper &records, raw_ostream &os,
                      const std::string &groupName);

bool emitPasssCAPIHeader(const llvm::RecordKeeper &records, raw_ostream &os,
                         const std::string &groupName);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CAPI_H_
