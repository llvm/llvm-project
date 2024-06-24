#include <cstdlib>
#include <cstring>
#include <string>

int main() {
  // Stack
  const char *stack_pointer = "stack_there_is_only_one_of_me";

  // Heap
  const std::string heap_string1("heap_there_is_exactly_two_of_me");
  const std::string heap_string2("heap_there_is_exactly_two_of_me");
  const char *heap_pointer1 = heap_string1.data();
  const char *heap_pointer2 = heap_string2.data();

  // Aligned Heap
  constexpr char aligned_string[] = "i_am_unaligned_string_on_the_heap";
  constexpr size_t len = sizeof(aligned_string) + 1;
  // Allocate memory aligned to 8-byte boundary
  void *aligned_string_ptr = aligned_alloc(8, len);
  memcpy(aligned_string_ptr, aligned_string, len);

  (void)stack_pointer;
  (void)heap_pointer1;
  (void)heap_pointer2;
  (void)aligned_string_ptr; // break here
  return 0;
}
