// Padding to fill the read-only segment to exactly one page (0x1000).
// The linker script places .text and .pad_ro in the first PT_LOAD,
// and .data in the second PT_LOAD, making them contiguous at a page
// boundary with different sizes.
int lib_func(int x) { return x * 2; }

// Writable data in the second segment
int lib_data = 100;
char lib_buf[256] = {1};
