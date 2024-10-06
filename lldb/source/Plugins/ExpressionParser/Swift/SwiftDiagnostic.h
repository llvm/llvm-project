//===-- SwiftDiagnostic.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_SwiftDiagnostic_h
#define lldb_SwiftDiagnostic_h

#include <vector>

#include "swift/AST/DiagnosticConsumer.h"

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include "lldb/Expression/DiagnosticManager.h"

namespace lldb_private {

class SwiftDiagnostic : public Diagnostic {
public:
  typedef std::vector<swift::DiagnosticInfo::FixIt> FixItList;

  static inline bool classof(const SwiftDiagnostic *) { return true; }
  static inline bool classof(const Diagnostic *diag) {
    return diag->getKind() == eDiagnosticOriginSwift;
  }

  SwiftDiagnostic(DiagnosticDetail detail, uint32_t compiler_id,
                  uint32_t buffer_id)
      : Diagnostic(eDiagnosticOriginSwift, compiler_id, detail),
        m_buffer_id(buffer_id) {}

  virtual ~SwiftDiagnostic() = default;

  bool HasFixIts() const override { return !m_fixit_vec.empty(); }

  void AddFixIt(const swift::DiagnosticInfo::FixIt &fixit) {
    m_fixit_vec.push_back(fixit);
  }

  const FixItList &FixIts() const { return m_fixit_vec; }

  uint32_t GetBufferID() const { return m_buffer_id; }

private:
  uint32_t m_buffer_id;
  FixItList m_fixit_vec;
};

} // namespace lldb_private
#endif /* lldb_SwiftDiagnostic_h */
