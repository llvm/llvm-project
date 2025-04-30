#include "tinyalloc.h"
#include <stdint.h>

#ifdef TA_DEBUG
extern void print_s(char *);
extern void print_i(size_t);
#else
#define print_s(X)
#define print_i(X)
#endif

typedef struct Block Block;

struct Block {
  void *addr;
  Block *next;
  size_t size;
};

typedef struct {
  Block *free; // first free block
  Block *used; // first used block
  Block *fresh; // first available blank block
  size_t top; // top free addr
} Heap;

static Heap *heap = NULL;
static const void *heap_limit = NULL;
static size_t heap_split_thresh;
static size_t heap_alignment;
static size_t heap_max_blocks;

/**
 * If compaction is enabled, inserts block
 * into free list, sorted by addr.
 * If disabled, add block has new head of
 * the free list.
 */
#pragma ns location host
static void insert_block(Block *block) {
#ifndef TA_DISABLE_COMPACT
  Block *ptr = heap->free;
  Block *prev = NULL;
  while (ptr != NULL) {
    if ((size_t)block->addr <= (size_t)ptr->addr) {
      print_s("insert");
      print_i((size_t)ptr);
      break;
    }
    prev = ptr;
    ptr = ptr->next;
  }
  if (prev != NULL) {
    if (ptr == NULL) {
      print_s("new tail");
    }
    prev->next = block;
  } else {
    print_s("new head");
    heap->free = block;
  }
  block->next = ptr;
#else
  block->next = heap->free;
  heap->free = block;
#endif
}

#ifndef TA_DISABLE_COMPACT
#pragma ns location host
static void release_blocks(Block *scan, Block *to) {
  Block *scan_next;
  while (scan != to) {
    print_s("release");
    print_i((size_t)scan);
    scan_next = scan->next;
    scan->next = heap->fresh;
    heap->fresh = scan;
    scan->addr = 0;
    scan->size = 0;
    scan = scan_next;
  }
}

static void compact() {
  Block *ptr = heap->free;
  Block *prev;
  Block *scan;
#pragma ns location host
  while (ptr != NULL) {
    prev = ptr;
    scan = ptr->next;
    while (scan != NULL &&
           (size_t)prev->addr + prev->size == (size_t)scan->addr) {
      print_s("merge");
      print_i((size_t)scan);
      prev = scan;
      scan = scan->next;
    }
    if (prev != ptr) {
      size_t new_size = (size_t)prev->addr - (size_t)ptr->addr + prev->size;
      print_s("new size");
      print_i(new_size);
      ptr->size = new_size;
      Block *next = prev->next;
      // make merged blocks available
      release_blocks(ptr->next, prev->next);
      // relink
      ptr->next = next;
    }
    ptr = ptr->next;
  }
}
#endif

#pragma ns location host
bool ta_init(void *base, const void *limit, const size_t heap_blocks,
             const size_t split_thresh, const size_t alignment) {
  heap = (Heap *)base;
  heap_limit = limit;
  heap_split_thresh = split_thresh;
  heap_alignment = alignment;
  heap_max_blocks = heap_blocks;

  heap->free = NULL;
  heap->used = NULL;
  heap->fresh = (Block *)(heap + 1);
  heap->top = (size_t)(heap->fresh + heap_blocks);

  Block *block = heap->fresh;
  size_t i = heap_max_blocks - 1;
  while (i--) {
    block->next = block + 1;
    block++;
  }
  block->next = NULL;
  return true;
}

#pragma ns location host
bool ta_free(void *free) {
  Block *block = heap->used;
  Block *prev = NULL;
#pragma ns location host
  while (block != NULL) {
    if (free == block->addr) {
      if (prev) {
        prev->next = block->next;
      } else {
        heap->used = block->next;
      }
      insert_block(block);
#ifndef TA_DISABLE_COMPACT
      compact();
#endif
      return true;
    }
    prev = block;
    block = block->next;
  }
  return false;
}

#pragma ns location host
static Block *alloc_block(size_t num) {
  Block *ptr = heap->free;
  Block *prev = NULL;
  size_t top = heap->top;
  num = (num + heap_alignment - 1) & -heap_alignment;
  while (ptr != NULL) {
    const int is_top = ((size_t)ptr->addr + ptr->size >= top) &&
                       ((size_t)ptr->addr + num <= (size_t)heap_limit);
    if (is_top || ptr->size >= num) {
      if (prev != NULL) {
        prev->next = ptr->next;
      } else {
        heap->free = ptr->next;
      }
      ptr->next = heap->used;
      heap->used = ptr;
      if (is_top) {
        print_s("resize top block");
        ptr->size = num;
        heap->top = (size_t)ptr->addr + num;
#ifndef TA_DISABLE_SPLIT
      } else if (heap->fresh != NULL) {
        size_t excess = ptr->size - num;
        if (excess >= heap_split_thresh) {
          ptr->size = num;
          Block *split = heap->fresh;
          heap->fresh = split->next;
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          split->addr = (void *)((size_t)ptr->addr + num);
          print_s("split");
          print_i((size_t)split->addr);
          split->size = excess;
          insert_block(split);
#ifndef TA_DISABLE_COMPACT
          compact();
#endif
        }
#endif
      }
      return ptr;
    }
    prev = ptr;
    ptr = ptr->next;
  }
  // no matching free blocks
  // see if any other blocks available
  size_t new_top = top + num;
  if (heap->fresh != NULL && new_top <= (size_t)heap_limit) {
    ptr = heap->fresh;
    heap->fresh = ptr->next;
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    ptr->addr = (void *)top;
    ptr->next = heap->used;
    ptr->size = num;
    heap->used = ptr;
    heap->top = new_top;
    return ptr;
  }
  return NULL;
}

#pragma ns location host
void *ta_alloc(size_t num) {
  Block *block = alloc_block(num);
  if (block != NULL) {
    return block->addr;
  }
  return NULL;
}

static void memclear(void *ptr, size_t num) {
  size_t *ptrw = (size_t *)ptr;
  size_t numw = (num & -sizeof(size_t)) / sizeof(size_t);
  while (numw--) {
    *ptrw++ = 0;
  }
  num &= (sizeof(size_t) - 1);
  uint8_t *ptrb = (uint8_t *)ptrw;
  while (num--) {
    *ptrb++ = 0;
  }
}

void *ta_calloc(size_t num, size_t size) {
  num *= size;
  Block *block = alloc_block(num);
  if (block != NULL) {
    memclear(block->addr, num);
    return block->addr;
  }
  return NULL;
}

static size_t count_blocks(Block *ptr) {
  size_t num = 0;
  while (ptr != NULL) {
    num++;
    ptr = ptr->next;
  }
  return num;
}

size_t ta_num_free() { return count_blocks(heap->free); }

size_t ta_num_used() { return count_blocks(heap->used); }

size_t ta_num_fresh() { return count_blocks(heap->fresh); }

bool ta_check() {
  return heap_max_blocks == ta_num_free() + ta_num_used() + ta_num_fresh();
}
