//===-- NativeThreadNetBSD.h ---------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeThreadNetBSD_H_
#define liblldb_NativeThreadNetBSD_H_

#include "lldb/Host/common/NativeThreadProtocol.h"

namespace lldb_private {
namespace process_netbsd {

class NativeProcessNetBSD;

class NativeThreadNetBSD : public NativeThreadProtocol {
  friend class NativeProcessNetBSD;

public:
  NativeThreadNetBSD(NativeProcessNetBSD *process, lldb::tid_t tid);
};

typedef std::shared_ptr<NativeThreadNetBSD> NativeThreadNetBSDSP;
} // namespace process_netbsd
} // namespace lldb_private

#endif // #ifndef liblldb_NativeThreadNetBSD_H_
