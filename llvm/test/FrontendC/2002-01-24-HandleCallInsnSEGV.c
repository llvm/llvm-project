// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

void *dlclose(void*);

void ap_os_dso_unload(void *handle)
{
    dlclose(handle);
    return;     /* This return triggers the bug: Weird */
}
