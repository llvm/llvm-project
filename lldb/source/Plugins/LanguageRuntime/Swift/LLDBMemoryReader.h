
#ifndef liblldb_LLDBMemoryReader_h_
#define liblldb_LLDBMemoryReader_h_

#include "SwiftLanguageRuntime.h"
#include "swift/Reflection/TypeLowering.h"
#include "llvm/Support/Memory.h"


namespace lldb_private {
class LLDBMemoryReader : public swift::remote::MemoryReader {
public:
  LLDBMemoryReader(Process &p, size_t max_read_amount = INT32_MAX)
      : m_process(p) {
    m_max_read_amount = max_read_amount;
  }

  virtual ~LLDBMemoryReader() = default;

  bool queryDataLayout(DataLayoutQueryType type, void *inBuffer,
                       void *outBuffer) override;

  swift::remote::RemoteAddress
  getSymbolAddress(const std::string &name) override;

  bool readBytes(swift::remote::RemoteAddress address, uint8_t *dest,
                 uint64_t size) override;

  bool readString(swift::remote::RemoteAddress address,
                  std::string &dest) override;

  void pushLocalBuffer(uint64_t local_buffer, uint64_t local_buffer_size);

  void popLocalBuffer(); 

private:
  Process &m_process;
  size_t m_max_read_amount;

  llvm::Optional<uint64_t> m_local_buffer;
  uint64_t m_local_buffer_size = 0;
};
} // namespace lldb_private
#endif
