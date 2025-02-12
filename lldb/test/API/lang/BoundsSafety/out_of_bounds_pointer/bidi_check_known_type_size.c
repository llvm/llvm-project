#include <assert.h>
#include <ptrcheck.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void noop() {};

typedef struct FAMS {
  int count;
  int buffer[__counted_by(count)];
} FAMS_t;

static FAMS_t *__bidi_indexable
get_next_fam_struct(FAMS_t *__bidi_indexable current) {
  uintptr_t current_size = sizeof(*current) + (sizeof(int) * current->count);
  uintptr_t next = ((uintptr_t)current) + current_size;
  uintptr_t num_bytes =
      ((uintptr_t)__ptr_upper_bound(current)) - ((uintptr_t)current);
  uintptr_t remaining_bytes = 0;
  if (num_bytes > current_size)
    remaining_bytes = num_bytes - current_size;
  return __unsafe_forge_bidi_indexable(FAMS_t *, next, remaining_bytes);
}

static FAMS_t *__bidi_indexable alloc_fams_buffer(size_t num_fams,
                                                  size_t num_elts_in_buffer) {
  const size_t buffer_size = sizeof(int) * num_elts_in_buffer;
  const size_t alloc_size = num_fams * (sizeof(FAMS_t) + buffer_size);
  FAMS_t *__bidi_indexable fams =
      __unsafe_forge_bidi_indexable(FAMS_t *, malloc(alloc_size), alloc_size);
  // Set the counts and zero init buffer
  FAMS_t *__bidi_indexable current = fams;
  for (size_t fam_num = 0; fam_num < num_fams; ++fam_num) {
    current->count = num_elts_in_buffer;
    // Don't zero-init due to rdar://102831737
    // memset(current->buffer, 0, buffer_size);
    current = get_next_fam_struct(current);
  }
  return fams;
}

void bidi_check_known_type_size() {
    char data[] = "fo";
    //--------------------------------------------------------------------------
    // ptr < lower bound
    //--------------------------------------------------------------------------
    char * __bidi_indexable oob_ptr_lower = data; // still in-bounds
    oob_ptr_lower -= 1; // break here: still in-bounds
    noop(); // break here: Now out-of-bounds

    //--------------------------------------------------------------------------
    // ptr + type size overflows
    //--------------------------------------------------------------------------
    int* __unsafe_indexable ptr_at_edge_of_addr_space = (int* __unsafe_indexable) UINTPTR_MAX;
    int * __bidi_indexable oob_ptr_plus_size_overflows = __unsafe_forge_bidi_indexable(int*, ptr_at_edge_of_addr_space, sizeof(int));
    noop(); // break here: oob_ptr_plus_size_overflows is out-of-bounds

    //--------------------------------------------------------------------------
    // ptr > upper bound
    //--------------------------------------------------------------------------
    char* __bidi_indexable oob_upper = data; // still in-bounds
    oob_upper += 1; // break here: still in-bounds
    oob_upper += 1; // break here: still in-bounds
    oob_upper += 1; // break here: still in-bounds
    noop(); // break here: Now out-of-bounds

    //--------------------------------------------------------------------------
    // Upper bound not aligned with type
    //--------------------------------------------------------------------------
    _Static_assert(sizeof(data) < sizeof(int), "oob-ness requirement failed");
    // oob_upper_type_pun is out-of-bounds because reading the sizeof(int) bytes
    // will read past the end of data.
    int* __bidi_indexable oob_upper_type_pun = (int* __bidi_indexable) data;
    noop(); // break here: oob_upper_type_pun is out-of-bounds

    //--------------------------------------------------------------------------
    // nullptr
    //--------------------------------------------------------------------------
    int* __bidi_indexable oob_null = 0x0;
    noop(); // break here: oob_null is out-of-bounds

    //--------------------------------------------------------------------------
    // Flexible array member
    //--------------------------------------------------------------------------
    FAMS_t *__bidi_indexable fams =
        alloc_fams_buffer(/*num_fams=*/2, /*num_elts_in_buffer=*/4);
    FAMS_t *__bidi_indexable fams0 = fams;
    noop(); // break here: fams0 in bounds

    FAMS_t *__bidi_indexable fams1 = get_next_fam_struct(fams0);
    noop(); // break here: fams1 in bounds

    FAMS_t *__bidi_indexable fams2 = get_next_fam_struct(fams1);
    noop(); // break here: fams2 is out-of-bounds

    free(fams);
    return;
}
