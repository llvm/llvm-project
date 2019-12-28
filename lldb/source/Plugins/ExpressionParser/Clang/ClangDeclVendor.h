//===-- ClangDeclVendor.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangDeclVendor_h_
#define liblldb_ClangDeclVendor_h_

#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/DeclVendor.h"

namespace lldb_private {

// A clang specialized extension to DeclVendor.
class ClangDeclVendor : public DeclVendor {
public:
  ClangDeclVendor(DeclVendorKind kind) : DeclVendor(kind) {}

  virtual ~ClangDeclVendor() {}

  using DeclVendor::FindDecls;

  uint32_t FindDecls(ConstString name, bool append, uint32_t max_matches,
                     std::vector<clang::NamedDecl *> &decls);

  static bool classof(const DeclVendor *vendor) {
    return vendor->GetKind() >= eClangDeclVendor &&
           vendor->GetKind() < eLastClangDeclVendor;
  }

private:
  DISALLOW_COPY_AND_ASSIGN(ClangDeclVendor);
};
} // namespace lldb_private

#endif
