#include <cstring>
#include <memory>
#include <string>

int main() {
  // Stack
  const char stack_pointer[] = "stack_there_is_only_one_of_me";

  // Heap
  // This test relies on std::string objects with size over 22 characters being
  // allocated on the heap.
  const std::string heap_string1("heap_there_is_exactly_two_of_me");
  const std::string heap_string2("heap_there_is_exactly_two_of_me");
  const char *heap_pointer1 = heap_string1.data();
  const char *heap_pointer2 = heap_string2.data();

  // Aligned Heap
  constexpr char aligned_string[] = "i_am_unaligned_string_on_the_heap";
  constexpr size_t buffer_size = 100;
  constexpr size_t len = sizeof(aligned_string) + 1;
  // Allocate memory aligned to 8-byte boundary
  void *aligned_string_ptr = new size_t[buffer_size];
  if (aligned_string_ptr == nullptr) {
    return -1;
  }
  // Zero out the memory
  memset(aligned_string_ptr, 0, buffer_size);

  // Align the pointer to a multiple of 8 bytes
  size_t size = buffer_size;
  aligned_string_ptr = std::align(8, len, aligned_string_ptr, size);

  // Copy the string to aligned memory
  memcpy(aligned_string_ptr, aligned_string, len);

  (void)stack_pointer;
  (void)heap_pointer1;
  (void)heap_pointer2; // break here
  return 0;
}
