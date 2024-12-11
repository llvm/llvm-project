#include <assert.h>
#include <ptrcheck.h>
#include <stdint.h>

static void noop() {}

void bidi_check_unknown_type_size() {
  char data[] = "fo";
  //--------------------------------------------------------------------------
  // ptr < lower bound
  //--------------------------------------------------------------------------
  void *__bidi_indexable oob_ptr_lower = data; // still in-bounds
  oob_ptr_lower -= 1;                          // break here: still in-bounds
  noop();                                      // break here: Now out-of-bounds

  //--------------------------------------------------------------------------
  // ptr + type size overflows
  //--------------------------------------------------------------------------
  void *__unsafe_indexable ptr_at_edge_of_addr_space =
      (void *__unsafe_indexable)UINTPTR_MAX;
  void *__bidi_indexable oob_ptr_plus_size_overflows =
      __unsafe_forge_bidi_indexable(void *, ptr_at_edge_of_addr_space,
                                    sizeof(int));
  noop(); // break here: oob_ptr_plus_size_overflows is now out-of-bounds

  //--------------------------------------------------------------------------
  // ptr > upper bound
  //--------------------------------------------------------------------------
  void *__bidi_indexable oob_upper = data; // still in-bounds
  oob_upper += 1;                          // break here: still in-bounds
  oob_upper += 1;                          // break here: still in-bounds
  oob_upper += 1;                          // break here: still in-bounds
  noop();                                  // break here: Now out-of-bounds

  //--------------------------------------------------------------------------
  // nullptr
  //--------------------------------------------------------------------------
  void *__bidi_indexable oob_null = 0x0;
  noop(); // break here: oob_null is out-of-bounds
  return;
}
