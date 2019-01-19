//===------ IslOstream.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// raw_ostream printers for isl C++ objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "isl/isl-noexceptions.h"
namespace polly {

#define ADD_OSTREAM_PRINTER(name)                                              \
  inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,                  \
                                       const name &Obj) {                      \
    OS << Obj.to_str();                                                        \
    return OS;                                                                 \
  }

ADD_OSTREAM_PRINTER(isl::aff)
ADD_OSTREAM_PRINTER(isl::ast_expr)
ADD_OSTREAM_PRINTER(isl::ast_node)
ADD_OSTREAM_PRINTER(isl::basic_map)
ADD_OSTREAM_PRINTER(isl::basic_set)
ADD_OSTREAM_PRINTER(isl::map)
ADD_OSTREAM_PRINTER(isl::set)
ADD_OSTREAM_PRINTER(isl::id)
ADD_OSTREAM_PRINTER(isl::multi_aff)
ADD_OSTREAM_PRINTER(isl::multi_pw_aff)
ADD_OSTREAM_PRINTER(isl::multi_union_pw_aff)
ADD_OSTREAM_PRINTER(isl::point)
ADD_OSTREAM_PRINTER(isl::pw_aff)
ADD_OSTREAM_PRINTER(isl::pw_multi_aff)
ADD_OSTREAM_PRINTER(isl::schedule)
ADD_OSTREAM_PRINTER(isl::schedule_node)
ADD_OSTREAM_PRINTER(isl::space)
ADD_OSTREAM_PRINTER(isl::union_access_info)
ADD_OSTREAM_PRINTER(isl::union_flow)
ADD_OSTREAM_PRINTER(isl::union_set)
ADD_OSTREAM_PRINTER(isl::union_map)
ADD_OSTREAM_PRINTER(isl::union_pw_aff)
ADD_OSTREAM_PRINTER(isl::union_pw_multi_aff)
} // namespace polly
