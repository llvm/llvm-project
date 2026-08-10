#include <linux/mman.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  // Note that we allocate memory here because if we used
  // stack or globals lldb might read it in the course of
  // running to the breakpoint. Before the test can look
  // for those reads.
  char *buf = mmap(0, page_size, PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  if (buf == MAP_FAILED)
    return 1;

  // Some known values to go in the corefile, since we cannot
  // write to corefile memory.
  buf[0] = 'L';
  buf[1] = 'L';
  buf[2] = 'D';
  buf[3] = 'B';

#define sign_ptr(ptr) __asm__ __volatile__("pacdza %0" : "=r"(ptr) : "r"(ptr))

  // Set top byte to something.
  char *buf_with_non_address = (char *)((size_t)buf | (size_t)0xff << 56);
  sign_ptr(buf_with_non_address);
  // Address is now:
  // <8 bit top byte tag><pointer signature><virtual address>

  // Uncomment this line to crash and generate a corefile.
  // Prints so we know what fixed address to look for in testing.
  // printf("buf: %p\n", buf);
  // printf("buf_with_non_address: %p\n", buf_with_non_address);
  // *(char*)0 = 0;

  return 0; // Set break point at this line.
}
