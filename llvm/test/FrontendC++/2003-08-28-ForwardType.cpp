// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

// Default placement versions of operator new.
#include <new>

void* operator new(size_t, void* __p) throw();


template<typename _CharT>
struct stdio_filebuf
{  stdio_filebuf();

};

extern stdio_filebuf<char> buf_cout;

void foo() {
  // Create stream buffers for the standard streams and use
  // those buffers without destroying and recreating the
  // streams.
  new (&buf_cout) stdio_filebuf<char>();

}
