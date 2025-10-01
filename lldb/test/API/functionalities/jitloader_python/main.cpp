#include <inttypes.h>
#include <stdio.h>

enum JITAction { 
  JIT_NOACTION = 0, 
  JIT_LOAD = 1, 
  JIT_UNLOAD = 2 
};

struct JITEntry
{
    struct JITEntry* next = nullptr;
    const char *path = nullptr;
    uint64_t address = 0;
};


JITEntry *g_entry_list = nullptr;

void jit_module_action(JITEntry *entry, JITAction action)
{
    printf("entry = %p, action = %i\n", (void *)entry, action);
}
// end GDB JIT interface


int main()
{
    // Create an empty JITEntry. The test case will set the path and address
    // with valid values at the first breakpoint. We build a "jit.out" binary
    // in the python test and we will load it at a address, so the test case
    // will calculate the path to the "jit.out" and the address to load it at
    // and set the values with some expressions.
    JITEntry entry;

    // Call the "jit_module_action" function to cause our JIT module to be
    // added to our target and loaded at an address.
    jit_module_action(&entry, JIT_LOAD); // Breakpoint 1
    printf("loaded module %s at %16.16" PRIx64 "\n", 
           entry.path, 
           entry.address);
    // Call the "jit_module_action" function to cause our JIT module to be
    // unloaded at an address.
    jit_module_action(&entry, JIT_UNLOAD); // Breakpoint 2
    printf("unloaded module %s" PRIx64 "\n", // Breakpoint 3
        entry.path);
 return 0;
}
