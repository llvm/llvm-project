#include <sys/mman.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

enum {
    kMmapSize = 0x1000,
    kMagicValue = 47,
};

void *address;
volatile sig_atomic_t signaled = 0;

void handler(int sig)
{
    signaled = 1;
    if (munmap(address, kMmapSize) != 0)
    {
        perror("munmap");
        _exit(5);
    }

    void* newaddr = mmap(address, kMmapSize, PROT_READ | PROT_WRITE,
            MAP_ANON | MAP_FIXED | MAP_PRIVATE, -1, 0);
    if (newaddr != address)
    {
        fprintf(stderr, "Newly mmaped address (%p) does not equal old address (%p).\n",
                newaddr, address);
        _exit(6);
    }
    *(int*)newaddr = kMagicValue;
}

int main()
{
    if (signal(SIGSEGV, handler) == SIG_ERR)
    {
        perror("signal");
        return 1;
    }

    address = mmap(NULL, kMmapSize, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (address == MAP_FAILED)
    {
        perror("mmap");
        return 2;
    }

    // This should first trigger a segfault. Our handler will make the memory readable and write
    // the magic value into memory.
    if (*(int*)address != kMagicValue)
        return 3;

    if (! signaled)
        return 4;

    return 0;
}
