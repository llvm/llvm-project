#include <inttypes.h>

// GDB JIT interface
enum JITAction { JIT_NOACTION, JIT_REGISTER_FN, JIT_UNREGISTER_FN };

struct JITCodeEntry {
  struct JITCodeEntry *next;
  struct JITCodeEntry *prev;
  const char *symfile_addr;
  uint64_t symfile_size;
};

struct JITDescriptor {
  uint32_t version;
  uint32_t action_flag;
  struct JITCodeEntry *relevant_entry;
  struct JITCodeEntry *first_entry;
};

// A single module whose first_entry is non-null but *misaligned*, i.e. it can
// never be a real jit_code_entry pointer. lldb must not assert when it walks
// the chain (see JITLoaderGDB.cpp, ReadJITEntry).
#define BAD ((struct JITCodeEntry *)0x0300cee3fb4002ffULL)
struct JITDescriptor __jit_debug_descriptor = {1, JIT_REGISTER_FN, BAD, BAD};

void __jit_debug_register_code() {}
// end GDB JIT interface

int main() {
  __jit_debug_register_code();
  return 0;
}
