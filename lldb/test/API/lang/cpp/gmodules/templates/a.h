#include "memory.h"

class MemoryBuffer { int buffer = 42; };

struct SrcBuffer {
  my_std::unique_ptr<MemoryBuffer> Buffer;
};
