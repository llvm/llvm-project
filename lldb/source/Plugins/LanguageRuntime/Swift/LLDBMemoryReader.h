//===-- LLDBMemoryReader.h --------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LLDBMemoryReader_h_
#define liblldb_LLDBMemoryReader_h_

#include "SwiftLanguageRuntime.h"

// We need to include ReflectionContext.h before TypeLowering.h to avoid
// conflicts between mach/machine.h and llvm/BinaryFormat/MachO.h.
#include "swift/RemoteInspection/ReflectionContext.h"
#include "swift/RemoteInspection/TypeLowering.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Memory.h"

namespace lldb_private {

std::unique_ptr<swift::SwiftObjectFileFormat>
GetSwiftObjectFileFormat(llvm::Triple::ObjectFormatType obj_format_type);

class LLDBMemoryReader;
class MemoryReaderLocalBufferHolder {
public:
  MemoryReaderLocalBufferHolder() : m_memory_reader(nullptr) {}
  MemoryReaderLocalBufferHolder(MemoryReaderLocalBufferHolder &&other)
      : m_memory_reader(other.m_memory_reader) {
    other.m_memory_reader = nullptr;
  }

  MemoryReaderLocalBufferHolder &operator=(MemoryReaderLocalBufferHolder &&other) {
    this->m_memory_reader = other.m_memory_reader;
    other.m_memory_reader = nullptr;
    return *this;
  }

  ~MemoryReaderLocalBufferHolder(); 
private:
  friend LLDBMemoryReader;

  MemoryReaderLocalBufferHolder(LLDBMemoryReader *memory_reader)
      : m_memory_reader(memory_reader) {}

  LLDBMemoryReader *m_memory_reader;
};

class LLDBMemoryReader : public swift::remote::MemoryReader {
public:
  /// Besides address space 0 (the DefaultAddressSpace), subclasses are free to
  /// use any address space for their own implementation purposes. LLDB uses
  /// this address space to track file addresses it sends to RemoteInspection.
  static constexpr uint8_t LLDBAddressSpace = 1;

  LLDBMemoryReader(Process &p,
                   std::function<swift::remote::RemoteAbsolutePointer(
                       swift::remote::RemoteAbsolutePointer)>
                       stripper,
                   size_t max_read_amount = INT32_MAX)
      : m_process(p), signedPointerStripper(stripper), m_range_module_map() {
    m_max_read_amount = max_read_amount;
  }

  virtual ~LLDBMemoryReader() = default;

  bool queryDataLayout(DataLayoutQueryType type, void *inBuffer,
                       void *outBuffer) override;

  swift::remote::RemoteAddress
  getSymbolAddress(const std::string &name) override;

  std::optional<swift::remote::RemoteAbsolutePointer>
  resolvePointerAsSymbol(swift::remote::RemoteAddress address) override;

  swift::remote::RemoteAbsolutePointer
  resolvePointer(swift::remote::RemoteAddress address,
                 uint64_t readValue) override;

  std::optional<swift::remote::RemoteAddress>
  resolveRemoteAddress(swift::remote::RemoteAddress address) const override;

  bool readBytes(swift::remote::RemoteAddress address, uint8_t *dest,
                 uint64_t size) override;

  bool readString(swift::remote::RemoteAddress address,
                  std::string &dest) override;

  MemoryReaderLocalBufferHolder pushLocalBuffer(uint64_t local_buffer, uint64_t local_buffer_size);

  /// Adds the module to the list of modules we're tracking using tagged
  /// addresses, so we can read memory from the file cache whenever possible.
  /// \return a pair of addresses indicating the start and end of this image in
  /// the tagged address space. None on failure.
  std::optional<std::pair<uint64_t, uint64_t>>
  addModuleToAddressMap(lldb::ModuleSP module, bool register_symbol_obj_file);

  /// Returns whether the filecache optimization is enabled or not.
  bool readMetadataFromFileCacheEnabled() const;

  /// Set or clear the progress callback.
  void SetProgressCallback(
      std::function<void(llvm::StringRef)> progress_callback = {}) {
    m_progress_callback = progress_callback;
  }

protected:
  bool readRemoteAddressImpl(swift::remote::RemoteAddress address,
                             swift::remote::RemoteAddress &out,
                             std::size_t integerSize) override;

private:
  friend MemoryReaderLocalBufferHolder;

  void popLocalBuffer();

  /// Gets the file address and module that were mapped to a given tagged
  /// address.
  std::optional<std::pair<uint64_t, lldb::ModuleSP>>
  getFileAddressAndModuleForTaggedAddress(
      swift::remote::RemoteAddress tagged_address) const;

  /// Resolves the address by either mapping a tagged address back to an LLDB
  /// Address with section + offset, or, in case the address is not tagged,
  /// constructing an LLDB address with just the offset.
  /// \return an Address with Section + offset  if we succesfully converted a
  /// tagged address back, an Address with just an offset if the address was not
  /// tagged, and None if the address was tagged but we couldn't convert it back
  /// to an Address.
  std::optional<Address>
  remoteAddressToLLDBAddress(swift::remote::RemoteAddress address) const;

  enum class ReadBytesResult { fail, success_from_file, success_from_memory };
  /// Implementation detail of readBytes. Returns a pair where the first element
  /// indicates whether the memory was read successfully, the second element
  /// indicates whether live memory was read.
  ReadBytesResult readBytesImpl(swift::remote::RemoteAddress address,
                                uint8_t *dest, uint64_t size);

  /// Reads memory from the symbol rich binary from the address into dest.
  /// \return true if it was able to successfully read memory.
  std::optional<Address> resolveRemoteAddressFromSymbolObjectFile(
      swift::remote::RemoteAddress address) const;

  Process &m_process;
  size_t m_max_read_amount;

  std::optional<uint64_t> m_local_buffer;
  uint64_t m_local_buffer_size = 0;

  std::function<swift::remote::RemoteAbsolutePointer(
      swift::remote::RemoteAbsolutePointer)>
      signedPointerStripper;

  /// LLDBMemoryReader prefers to read reflection metadata from the
  /// binary on disk, which is faster than reading it out of process
  /// memory, especially when debugging remotely.  To achieve this LLDB
  /// registers virtual addresses starting at (0x0 &
  /// LLDB_VIRTUAL_ADDRESS_BIT) with ReflectionContext.  Sorted by
  /// virtual address, m_lldb_virtual_address_map stores each
  /// lldb::Module and the first virtual address after the end of that
  /// module's virtual address space.
  std::vector<std::pair<uint64_t, lldb::ModuleSP>> m_range_module_map;

  /// The set of modules where we should read memory from the symbol file's
  /// object file instead of the main object file.
  llvm::SmallSet<lldb::ModuleSP, 8> m_modules_with_metadata_in_symbol_obj_file;
  /// A callback to update a progress event.
  std::function<void(llvm::StringRef)> m_progress_callback;
};
} // namespace lldb_private
#endif
