#include <inttypes.h>

// GDB JIT interface
enum JITAction { JIT_NOACTION, JIT_REGISTER_FN, JIT_UNREGISTER_FN };

struct JITCodeEntry
{
    struct JITCodeEntry* next;
    struct JITCodeEntry* prev;
    const char *symfile_addr;
    uint64_t symfile_size;
};

struct JITDescriptor
{
    uint32_t version;
    uint32_t action_flag;
    struct JITCodeEntry* relevant_entry;
    struct JITCodeEntry* first_entry;
};

struct JITDescriptor __jit_debug_descriptor = { 1, JIT_NOACTION, 0, 0 };

void __jit_debug_register_code()
{
}
// end GDB JIT interface

struct JITCodeEntry entry;

int main()
{
    // Create a code entry with a bogus size
    entry.next = entry.prev = 0;
    entry.symfile_addr = (char *)&entry;
    entry.symfile_size = (uint64_t)47<<32;

    __jit_debug_descriptor.relevant_entry = __jit_debug_descriptor.first_entry = &entry;
    __jit_debug_descriptor.action_flag = JIT_REGISTER_FN;

    __jit_debug_register_code();

    return 0;
}
