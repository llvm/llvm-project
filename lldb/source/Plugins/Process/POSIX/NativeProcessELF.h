//===-- NativeProcessELF.h ------------------------------------ -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessELF_H_
#define liblldb_NativeProcessELF_H_

#include "Plugins/Process/Utility/AuxVector.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "llvm/BinaryFormat/ELF.h"

namespace lldb_private {

/// \class NativeProcessELF
/// Abstract class that extends \a NativeProcessProtocol with ELF specific
/// logic. Meant to be subclassed by ELF based NativeProcess* implementations.
class NativeProcessELF : public NativeProcessProtocol {
  using NativeProcessProtocol::NativeProcessProtocol;

protected:
  template <typename T> struct ELFLinkMap {
    T l_addr;
    T l_name;
    T l_ld;
    T l_next;
    T l_prev;
  };

  llvm::Optional<uint64_t> GetAuxValue(enum AuxVector::EntryType type);

  lldb::addr_t GetSharedLibraryInfoAddress() override;

  template <typename ELF_EHDR, typename ELF_PHDR, typename ELF_DYN>
  lldb::addr_t GetELFImageInfoAddress();

  std::unique_ptr<AuxVector> m_aux_vector;
  llvm::Optional<lldb::addr_t> m_shared_library_info_addr;
};

} // namespace lldb_private

#endif