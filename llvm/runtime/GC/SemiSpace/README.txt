//===----------------------------------------------------------------------===//

Possible enhancement: If a collection cycle happens and the heap is not
compacted very much (say less than 25% of the allocated memory was freed), the
memory regions should be doubled in size.
