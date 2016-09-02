//===-- MinidumpParser.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MinidumpParser_h_
#define liblldb_MinidumpParser_h_

// Project includes
#include "MinidumpTypes.h"

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Error.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

// C includes

// C++ includes
#include <cstring>
#include <unordered_map>


namespace lldb_private
{

namespace minidump
{

class MinidumpParser
{
public:
    static llvm::Optional<MinidumpParser>
    Create(const lldb::DataBufferSP &data_buf_sp);

    lldb::offset_t
    GetByteSize();

    llvm::Optional<llvm::ArrayRef<uint8_t>>
    GetStream(MinidumpStreamType stream_type);

    llvm::Optional<std::vector<const MinidumpThread *>>
    GetThreads();

    const MinidumpSystemInfo *
    GetSystemInfo();

    ArchSpec
    GetArchitecture();

    const MinidumpMiscInfo *
    GetMiscInfo();

private:
    lldb::DataBufferSP m_data_sp;
    const MinidumpHeader *m_header;
    llvm::DenseMap<uint32_t, MinidumpLocationDescriptor> m_directory_map;

    MinidumpParser(const lldb::DataBufferSP &data_buf_sp, const MinidumpHeader *header,
                   const llvm::DenseMap<uint32_t, MinidumpLocationDescriptor> &directory_map);
};

} // namespace minidump
} // namespace lldb_private
#endif // liblldb_MinidumpParser_h_
