
#ifndef liblldb_LLDBMemoryReader_h_
#define liblldb_LLDBMemoryReader_h_


#include "SwiftLanguageRuntime.h"

// We need to include ReflectionContext.h before TypeLowering.h to avoid
// conflicts between mach/machine.h and llvm/BinaryFormat/MachO.h.
#include "swift/Reflection/ReflectionContext.h"
#include "swift/Reflection/TypeLowering.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Memory.h"


namespace lldb_private {
class LLDBMemoryReader : public swift::remote::MemoryReader {
public:
  LLDBMemoryReader(Process &p, size_t max_read_amount = INT32_MAX)
      : m_process(p), m_range_module_map() {
    m_max_read_amount = max_read_amount;
  }

  virtual ~LLDBMemoryReader() = default;

  bool queryDataLayout(DataLayoutQueryType type, void *inBuffer,
                       void *outBuffer) override;

  swift::remote::RemoteAddress
  getSymbolAddress(const std::string &name) override;

  llvm::Optional<swift::remote::RemoteAbsolutePointer>
  resolvePointerAsSymbol(swift::remote::RemoteAddress address) override;

  swift::remote::RemoteAbsolutePointer
  resolvePointer(swift::remote::RemoteAddress address,
                 uint64_t readValue) override;

  bool readBytes(swift::remote::RemoteAddress address, uint8_t *dest,
                 uint64_t size) override;

  bool readString(swift::remote::RemoteAddress address,
                  std::string &dest) override;

  void pushLocalBuffer(uint64_t local_buffer, uint64_t local_buffer_size);

  void popLocalBuffer();

  /// Adds the module to the list of modules we're tracking using tagged
  /// addresses, so we can read memory from the file cache whenever possible.
  /// \return a pair of addresses indicating the start and end of this image in
  /// the tagged address space. None on failure.
  llvm::Optional<std::pair<uint64_t, uint64_t>>
  addModuleToAddressMap(lldb::ModuleSP module, bool register_symbol_obj_file);

  /// Returns whether the filecache optimization is enabled or not.
  bool readMetadataFromFileCacheEnabled() const;

private:
  /// Gets the file address and module that were mapped to a given tagged
  /// address.
  llvm::Optional<std::pair<uint64_t, lldb::ModuleSP>>
  getFileAddressAndModuleForTaggedAddress(uint64_t tagged_address) const;

  /// Resolves the address by either mapping a tagged address back to an LLDB 
  /// Address with section + offset, or, in case the address is not tagged, 
  /// constructing an LLDB address with just the offset.
  /// \return an Address with Section + offset  if we succesfully converted a tagged
  /// address back, an Address with just an offset if the address was not tagged,
  /// and None if the address was tagged but we couldn't convert it back to an 
  /// Address.
  llvm::Optional<Address> resolveRemoteAddress(uint64_t address) const;

 /// Reads memory from the symbol rich binary from the address into dest.
 /// \return true if it was able to successfully read memory.
llvm::Optional<Address> resolveRemoteAddressFromSymbolObjectFile(uint64_t address) const;


private:
  Process &m_process;
  size_t m_max_read_amount;

  llvm::Optional<uint64_t> m_local_buffer;
  uint64_t m_local_buffer_size = 0;

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
  llvm::SmallSet<lldb::ModuleSP, 8>
   m_modules_with_metadata_in_symbol_obj_file;

  /// The bit used to tag LLDB's virtual addresses as such. See \c
  /// m_range_module_map.
  const static uint64_t LLDB_FILE_ADDRESS_BIT = 0x2000000000000000;
  static_assert(LLDB_FILE_ADDRESS_BIT & SWIFT_ABI_X86_64_SWIFT_SPARE_BITS_MASK,
    "LLDB file address bit not in spare bits mask!");
  static_assert(LLDB_FILE_ADDRESS_BIT & SWIFT_ABI_ARM64_SWIFT_SPARE_BITS_MASK,
    "LLDB file address bit not in spare bits mask!");

};
} // namespace lldb_private
#endif
