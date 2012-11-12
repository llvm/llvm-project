//===-- ProcessDataAllocator.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Expression/ProcessDataAllocator.h"

using namespace lldb_private;

void
ProcessDataAllocator::Dump(Stream &stream)
{
    size_t data_size = m_stream_string.GetSize();
    
    if (!m_allocation)
        return;
    
    lldb::DataBufferSP data(new DataBufferHeap(data_size, 0));    
    
    Error error;
    if (m_process.ReadMemory (m_allocation, data->GetBytes(), data_size, error) != data_size)
        return;
    
    DataExtractor extractor(data, m_process.GetByteOrder(), m_process.GetAddressByteSize());
    
    extractor.Dump(&stream,                         // stream
                   0,                               // offset
                   lldb::eFormatBytesWithASCII,     // format
                   1,                               // byte size of individual entries
                   data_size,                       // number of entries
                   16,                              // entries per line
                   m_allocation,                    // address to print
                   0,                               // bit size (bitfields only; 0 means ignore)
                   0);                              // bit alignment (bitfields only; 0 means ignore)
    
    stream.PutChar('\n');
}
