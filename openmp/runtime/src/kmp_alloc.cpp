/*
 * kmp_alloc.cpp -- private/shared dynamic memory allocation and management
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_io.h"
#include "kmp_wrapper_malloc.h"

// Disable bget when it is not used
#if KMP_USE_BGET

/* Thread private buffer management code */

typedef int (*bget_compact_t)(size_t, int);
typedef void *(*bget_acquire_t)(size_t);
typedef void (*bget_release_t)(void *);

/* NOTE: bufsize must be a signed datatype */

#if KMP_OS_WINDOWS
#if KMP_ARCH_X86 || KMP_ARCH_ARM
typedef kmp_int32 bufsize;
#else
typedef kmp_int64 bufsize;
#endif
#else
typedef ssize_t bufsize;
#endif // KMP_OS_WINDOWS

/* The three modes of operation are, fifo search, lifo search, and best-fit */

typedef enum bget_mode {
  bget_mode_fifo = 0,
  bget_mode_lifo = 1,
  bget_mode_best = 2
} bget_mode_t;

static void bpool(kmp_info_t *th, void *buffer, bufsize len);
static void *bget(kmp_info_t *th, bufsize size);
static void *bgetz(kmp_info_t *th, bufsize size);
static void *bgetr(kmp_info_t *th, void *buffer, bufsize newsize);
static void brel(kmp_info_t *th, void *buf);
static void bectl(kmp_info_t *th, bget_compact_t compact,
                  bget_acquire_t acquire, bget_release_t release,
                  bufsize pool_incr);

/* BGET CONFIGURATION */
/* Buffer allocation size quantum: all buffers allocated are a
   multiple of this size.  This MUST be a power of two. */

/* On IA-32 architecture with  Linux* OS, malloc() does not
   ensure 16 byte alignment */

#if KMP_ARCH_X86 || !KMP_HAVE_QUAD

#define SizeQuant 8
#define AlignType double

#else

#define SizeQuant 16
#define AlignType _Quad

#endif

// Define this symbol to enable the bstats() function which calculates the
// total free space in the buffer pool, the largest available buffer, and the
// total space currently allocated.
#define BufStats 1

#ifdef KMP_DEBUG

// Define this symbol to enable the bpoold() function which dumps the buffers
// in a buffer pool.
#define BufDump 1

// Define this symbol to enable the bpoolv() function for validating a buffer
// pool.
#define BufValid 1

// Define this symbol to enable the bufdump() function which allows dumping the
// contents of an allocated or free buffer.
#define DumpData 1

#ifdef NOT_USED_NOW

// Wipe free buffers to a guaranteed pattern of garbage to trip up miscreants
// who attempt to use pointers into released buffers.
#define FreeWipe 1

// Use a best fit algorithm when searching for space for an allocation request.
// This uses memory more efficiently, but allocation will be much slower.
#define BestFit 1

#endif /* NOT_USED_NOW */
#endif /* KMP_DEBUG */

static bufsize bget_bin_size[] = {
    0,
    //    1 << 6,    /* .5 Cache line */
    1 << 7, /* 1 Cache line, new */
    1 << 8, /* 2 Cache lines */
    1 << 9, /* 4 Cache lines, new */
    1 << 10, /* 8 Cache lines */
    1 << 11, /* 16 Cache lines, new */
    1 << 12, 1 << 13, /* new */
    1 << 14, 1 << 15, /* new */
    1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, /*  1MB */
    1 << 21, /*  2MB */
    1 << 22, /*  4MB */
    1 << 23, /*  8MB */
    1 << 24, /* 16MB */
    1 << 25, /* 32MB */
};

#define MAX_BGET_BINS (int)(sizeof(bget_bin_size) / sizeof(bufsize))

struct bfhead;

//  Declare the interface, including the requested buffer size type, bufsize.

/* Queue links */
typedef struct qlinks {
  struct bfhead *flink; /* Forward link */
  struct bfhead *blink; /* Backward link */
} qlinks_t;

/* Header in allocated and free buffers */
typedef struct bhead2 {
  kmp_info_t *bthr; /* The thread which owns the buffer pool */
  bufsize prevfree; /* Relative link back to previous free buffer in memory or
                       0 if previous buffer is allocated.  */
  bufsize bsize; /* Buffer size: positive if free, negative if allocated. */
} bhead2_t;

/* Make sure the bhead structure is a multiple of SizeQuant in size. */
typedef union bhead {
  KMP_ALIGN(SizeQuant)
  AlignType b_align;
  char b_pad[sizeof(bhead2_t) + (SizeQuant - (sizeof(bhead2_t) % SizeQuant))];
  bhead2_t bb;
} bhead_t;
#define BH(p) ((bhead_t *)(p))

/*  Header in directly allocated buffers (by acqfcn) */
typedef struct bdhead {
  bufsize tsize; /* Total size, including overhead */
  bhead_t bh; /* Common header */
} bdhead_t;
#define BDH(p) ((bdhead_t *)(p))

/* Header in free buffers */
typedef struct bfhead {
  bhead_t bh; /* Common allocated/free header */
  qlinks_t ql; /* Links on free list */
} bfhead_t;
#define BFH(p) ((bfhead_t *)(p))

typedef struct thr_data {
  bfhead_t freelist[MAX_BGET_BINS];
#if BufStats
  size_t totalloc; /* Total space currently allocated */
  long numget, numrel; /* Number of bget() and brel() calls */
  long numpblk; /* Number of pool blocks */
  long numpget, numprel; /* Number of block gets and rels */
  long numdget, numdrel; /* Number of direct gets and rels */
#endif /* BufStats */

  /* Automatic expansion block management functions */
  bget_compact_t compfcn;
  bget_acquire_t acqfcn;
  bget_release_t relfcn;

  bget_mode_t mode; /* what allocation mode to use? */

  bufsize exp_incr; /* Expansion block size */
  bufsize pool_len; /* 0: no bpool calls have been made
                       -1: not all pool blocks are the same size
                       >0: (common) block size for all bpool calls made so far
                    */
  bfhead_t *last_pool; /* Last pool owned by this thread (delay deallocation) */
} thr_data_t;

/*  Minimum allocation quantum: */
#define QLSize (sizeof(qlinks_t))
#define SizeQ ((SizeQuant > QLSize) ? SizeQuant : QLSize)
#define MaxSize                                                                \
  (bufsize)(                                                                   \
      ~(((bufsize)(1) << (sizeof(bufsize) * CHAR_BIT - 1)) | (SizeQuant - 1)))
// Maximum for the requested size.

/* End sentinel: value placed in bsize field of dummy block delimiting
   end of pool block.  The most negative number which will  fit  in  a
   bufsize, defined in a way that the compiler will accept. */

#define ESent                                                                  \
  ((bufsize)(-(((((bufsize)1) << ((int)sizeof(bufsize) * 8 - 2)) - 1) * 2) - 2))

/* Thread Data management routines */
static int bget_get_bin(bufsize size) {
  // binary chop bins
  int lo = 0, hi = MAX_BGET_BINS - 1;

  KMP_DEBUG_ASSERT(size > 0);

  while ((hi - lo) > 1) {
    int mid = (lo + hi) >> 1;
    if (size < bget_bin_size[mid])
      hi = mid - 1;
    else
      lo = mid;
  }

  KMP_DEBUG_ASSERT((lo >= 0) && (lo < MAX_BGET_BINS));

  return lo;
}

static void set_thr_data(kmp_info_t *th) {
  int i;
  thr_data_t *data;

  data = (thr_data_t *)((!th->th.th_local.bget_data)
                            ? __kmp_allocate(sizeof(*data))
                            : th->th.th_local.bget_data);

  memset(data, '\0', sizeof(*data));

  for (i = 0; i < MAX_BGET_BINS; ++i) {
    data->freelist[i].ql.flink = &data->freelist[i];
    data->freelist[i].ql.blink = &data->freelist[i];
  }

  th->th.th_local.bget_data = data;
  th->th.th_local.bget_list = 0;
#if !USE_CMP_XCHG_FOR_BGET
#ifdef USE_QUEUING_LOCK_FOR_BGET
  __kmp_init_lock(&th->th.th_local.bget_lock);
#else
  __kmp_init_bootstrap_lock(&th->th.th_local.bget_lock);
#endif /* USE_LOCK_FOR_BGET */
#endif /* ! USE_CMP_XCHG_FOR_BGET */
}

static thr_data_t *get_thr_data(kmp_info_t *th) {
  thr_data_t *data;

  data = (thr_data_t *)th->th.th_local.bget_data;

  KMP_DEBUG_ASSERT(data != 0);

  return data;
}

/* Walk the free list and release the enqueued buffers */
static void __kmp_bget_dequeue(kmp_info_t *th) {
  void *p = TCR_SYNC_PTR(th->th.th_local.bget_list);

  if (p != 0) {
#if USE_CMP_XCHG_FOR_BGET
    {
      volatile void *old_value = TCR_SYNC_PTR(th->th.th_local.bget_list);
      while (!KMP_COMPARE_AND_STORE_PTR(&th->th.th_local.bget_list,
                                        CCAST(void *, old_value), nullptr)) {
        KMP_CPU_PAUSE();
        old_value = TCR_SYNC_PTR(th->th.th_local.bget_list);
      }
      p = CCAST(void *, old_value);
    }
#else /* ! USE_CMP_XCHG_FOR_BGET */
#ifdef USE_QUEUING_LOCK_FOR_BGET
    __kmp_acquire_lock(&th->th.th_local.bget_lock, __kmp_gtid_from_thread(th));
#else
    __kmp_acquire_bootstrap_lock(&th->th.th_local.bget_lock);
#endif /* USE_QUEUING_LOCK_FOR_BGET */

    p = (void *)th->th.th_local.bget_list;
    th->th.th_local.bget_list = 0;

#ifdef USE_QUEUING_LOCK_FOR_BGET
    __kmp_release_lock(&th->th.th_local.bget_lock, __kmp_gtid_from_thread(th));
#else
    __kmp_release_bootstrap_lock(&th->th.th_local.bget_lock);
#endif
#endif /* USE_CMP_XCHG_FOR_BGET */

    /* Check again to make sure the list is not empty */
    while (p != 0) {
      void *buf = p;
      bfhead_t *b = BFH(((char *)p) - sizeof(bhead_t));

      KMP_DEBUG_ASSERT(b->bh.bb.bsize != 0);
      KMP_DEBUG_ASSERT(((kmp_uintptr_t)TCR_PTR(b->bh.bb.bthr) & ~1) ==
                       (kmp_uintptr_t)th); // clear possible mark
      KMP_DEBUG_ASSERT(b->ql.blink == 0);

      p = (void *)b->ql.flink;

      brel(th, buf);
    }
  }
}

/* Chain together the free buffers by using the thread owner field */
static void __kmp_bget_enqueue(kmp_info_t *th, void *buf
#ifdef USE_QUEUING_LOCK_FOR_BGET
                               ,
                               kmp_int32 rel_gtid
#endif
) {
  bfhead_t *b = BFH(((char *)buf) - sizeof(bhead_t));

  KMP_DEBUG_ASSERT(b->bh.bb.bsize != 0);
  KMP_DEBUG_ASSERT(((kmp_uintptr_t)TCR_PTR(b->bh.bb.bthr) & ~1) ==
                   (kmp_uintptr_t)th); // clear possible mark

  b->ql.blink = 0;

  KC_TRACE(10, ("__kmp_bget_enqueue: moving buffer to T#%d list\n",
                __kmp_gtid_from_thread(th)));

#if USE_CMP_XCHG_FOR_BGET
  {
    volatile void *old_value = TCR_PTR(th->th.th_local.bget_list);
    /* the next pointer must be set before setting bget_list to buf to avoid
       exposing a broken list to other threads, even for an instant. */
    b->ql.flink = BFH(CCAST(void *, old_value));

    while (!KMP_COMPARE_AND_STORE_PTR(&th->th.th_local.bget_list,
                                      CCAST(void *, old_value), buf)) {
      KMP_CPU_PAUSE();
      old_value = TCR_PTR(th->th.th_local.bget_list);
      /* the next pointer must be set before setting bget_list to buf to avoid
         exposing a broken list to other threads, even for an instant. */
      b->ql.flink = BFH(CCAST(void *, old_value));
    }
  }
#else /* ! USE_CMP_XCHG_FOR_BGET */
#ifdef USE_QUEUING_LOCK_FOR_BGET
  __kmp_acquire_lock(&th->th.th_local.bget_lock, rel_gtid);
#else
  __kmp_acquire_bootstrap_lock(&th->th.th_local.bget_lock);
#endif

  b->ql.flink = BFH(th->th.th_local.bget_list);
  th->th.th_local.bget_list = (void *)buf;

#ifdef USE_QUEUING_LOCK_FOR_BGET
  __kmp_release_lock(&th->th.th_local.bget_lock, rel_gtid);
#else
  __kmp_release_bootstrap_lock(&th->th.th_local.bget_lock);
#endif
#endif /* USE_CMP_XCHG_FOR_BGET */
}

/* insert buffer back onto a new freelist */
static void __kmp_bget_insert_into_freelist(thr_data_t *thr, bfhead_t *b) {
  int bin;

  KMP_DEBUG_ASSERT(((size_t)b) % SizeQuant == 0);
  KMP_DEBUG_ASSERT(b->bh.bb.bsize % SizeQuant == 0);

  bin = bget_get_bin(b->bh.bb.bsize);

  KMP_DEBUG_ASSERT(thr->freelist[bin].ql.blink->ql.flink ==
                   &thr->freelist[bin]);
  KMP_DEBUG_ASSERT(thr->freelist[bin].ql.flink->ql.blink ==
                   &thr->freelist[bin]);

  b->ql.flink = &thr->freelist[bin];
  b->ql.blink = thr->freelist[bin].ql.blink;

  thr->freelist[bin].ql.blink = b;
  b->ql.blink->ql.flink = b;
}

/* unlink the buffer from the old freelist */
static void __kmp_bget_remove_from_freelist(bfhead_t *b) {
  KMP_DEBUG_ASSERT(b->ql.blink->ql.flink == b);
  KMP_DEBUG_ASSERT(b->ql.flink->ql.blink == b);

  b->ql.blink->ql.flink = b->ql.flink;
  b->ql.flink->ql.blink = b->ql.blink;
}

/*  GET STATS -- check info on free list */
static void bcheck(kmp_info_t *th, bufsize *max_free, bufsize *total_free) {
  thr_data_t *thr = get_thr_data(th);
  int bin;

  *total_free = *max_free = 0;

  for (bin = 0; bin < MAX_BGET_BINS; ++bin) {
    bfhead_t *b, *best;

    best = &thr->freelist[bin];
    b = best->ql.flink;

    while (b != &thr->freelist[bin]) {
      *total_free += (b->bh.bb.bsize - sizeof(bhead_t));
      if ((best == &thr->freelist[bin]) || (b->bh.bb.bsize < best->bh.bb.bsize))
        best = b;

      /* Link to next buffer */
      b = b->ql.flink;
    }

    if (*max_free < best->bh.bb.bsize)
      *max_free = best->bh.bb.bsize;
  }

  if (*max_free > (bufsize)sizeof(bhead_t))
    *max_free -= sizeof(bhead_t);
}

/*  BGET  --  Allocate a buffer.  */
static void *bget(kmp_info_t *th, bufsize requested_size) {
  thr_data_t *thr = get_thr_data(th);
  bufsize size = requested_size;
  bfhead_t *b;
  void *buf;
  int compactseq = 0;
  int use_blink = 0;
  /* For BestFit */
  bfhead_t *best;

  if (size < 0 || size + sizeof(bhead_t) > MaxSize) {
    return NULL;
  }

  __kmp_bget_dequeue(th); /* Release any queued buffers */

  if (size < (bufsize)SizeQ) { // Need at least room for the queue links.
    size = SizeQ;
  }
#if defined(SizeQuant) && (SizeQuant > 1)
  size = (size + (SizeQuant - 1)) & (~(SizeQuant - 1));
#endif

  size += sizeof(bhead_t); // Add overhead in allocated buffer to size required.
  KMP_DEBUG_ASSERT(size >= 0);
  KMP_DEBUG_ASSERT(size % SizeQuant == 0);

  use_blink = (thr->mode == bget_mode_lifo);

  /* If a compact function was provided in the call to bectl(), wrap
     a loop around the allocation process  to  allow  compaction  to
     intervene in case we don't find a suitable buffer in the chain. */

  for (;;) {
    int bin;

    for (bin = bget_get_bin(size); bin < MAX_BGET_BINS; ++bin) {
      /* Link to next buffer */
      b = (use_blink ? thr->freelist[bin].ql.blink
                     : thr->freelist[bin].ql.flink);

      if (thr->mode == bget_mode_best) {
        best = &thr->freelist[bin];

        /* Scan the free list searching for the first buffer big enough
           to hold the requested size buffer. */
        while (b != &thr->freelist[bin]) {
          if (b->bh.bb.bsize >= (bufsize)size) {
            if ((best == &thr->freelist[bin]) ||
                (b->bh.bb.bsize < best->bh.bb.bsize)) {
              best = b;
            }
          }

          /* Link to next buffer */
          b = (use_blink ? b->ql.blink : b->ql.flink);
        }
        b = best;
      }

      while (b != &thr->freelist[bin]) {
        if ((bufsize)b->bh.bb.bsize >= (bufsize)size) {

          // Buffer is big enough to satisfy the request. Allocate it to the
          // caller. We must decide whether the buffer is large enough to split
          // into the part given to the caller and a free buffer that remains
          // on the free list, or whether the entire buffer should be removed
          // from the free list and given to the caller in its entirety. We
          // only split the buffer if enough room remains for a header plus the
          // minimum quantum of allocation.
          if ((b->bh.bb.bsize - (bufsize)size) >
              (bufsize)(SizeQ + (sizeof(bhead_t)))) {
            bhead_t *ba, *bn;

            ba = BH(((char *)b) + (b->bh.bb.bsize - (bufsize)size));
            bn = BH(((char *)ba) + size);

            KMP_DEBUG_ASSERT(bn->bb.prevfree == b->bh.bb.bsize);

            /* Subtract size from length of free block. */
            b->bh.bb.bsize -= (bufsize)size;

            /* Link allocated buffer to the previous free buffer. */
            ba->bb.prevfree = b->bh.bb.bsize;

            /* Plug negative size into user buffer. */
            ba->bb.bsize = -size;

            /* Mark this buffer as owned by this thread. */
            TCW_PTR(ba->bb.bthr,
                    th); // not an allocated address (do not mark it)
            /* Mark buffer after this one not preceded by free block. */
            bn->bb.prevfree = 0;

            // unlink buffer from old freelist, and reinsert into new freelist
            __kmp_bget_remove_from_freelist(b);
            __kmp_bget_insert_into_freelist(thr, b);
#if BufStats
            thr->totalloc += (size_t)size;
            thr->numget++; /* Increment number of bget() calls */
#endif
            buf = (void *)((((char *)ba) + sizeof(bhead_t)));
            KMP_DEBUG_ASSERT(((size_t)buf) % SizeQuant == 0);
            return buf;
          } else {
            bhead_t *ba;

            ba = BH(((char *)b) + b->bh.bb.bsize);

            KMP_DEBUG_ASSERT(ba->bb.prevfree == b->bh.bb.bsize);

            /* The buffer isn't big enough to split.  Give  the  whole
               shebang to the caller and remove it from the free list. */

            __kmp_bget_remove_from_freelist(b);
#if BufStats
            thr->totalloc += (size_t)b->bh.bb.bsize;
            thr->numget++; /* Increment number of bget() calls */
#endif
            /* Negate size to mark buffer allocated. */
            b->bh.bb.bsize = -(b->bh.bb.bsize);

            /* Mark this buffer as owned by this thread. */
            TCW_PTR(ba->bb.bthr, th); // not an allocated address (do not mark)
            /* Zero the back pointer in the next buffer in memory
               to indicate that this buffer is allocated. */
            ba->bb.prevfree = 0;

            /* Give user buffer starting at queue links. */
            buf = (void *)&(b->ql);
            KMP_DEBUG_ASSERT(((size_t)buf) % SizeQuant == 0);
            return buf;
          }
        }

        /* Link to next buffer */
        b = (use_blink ? b->ql.blink : b->ql.flink);
      }
    }

    /* We failed to find a buffer. If there's a compact function defined,
       notify it of the size requested. If it returns TRUE, try the allocation
       again. */

    if ((thr->compfcn == 0) || (!(*thr->compfcn)(size, ++compactseq))) {
      break;
    }
  }

  /* No buffer available with requested size free. */

  /* Don't give up yet -- look in the reserve supply. */
  if (thr->acqfcn != 0) {
    if (size > (bufsize)(thr->exp_incr - sizeof(bhead_t))) {
      /* Request is too large to fit in a single expansion block.
         Try to satisfy it by a direct buffer acquisition. */
      bdhead_t *bdh;

      size += sizeof(bdhead_t) - sizeof(bhead_t);

      KE_TRACE(10, ("%%%%%% MALLOC( %d )\n", (int)size));

      /* richryan */
      bdh = BDH((*thr->acqfcn)((bufsize)size));
      if (bdh != NULL) {

        // Mark the buffer special by setting size field of its header to zero.
        bdh->bh.bb.bsize = 0;

        /* Mark this buffer as owned by this thread. */
        TCW_PTR(bdh->bh.bb.bthr, th); // don't mark buffer as allocated,
        // because direct buffer never goes to free list
        bdh->bh.bb.prevfree = 0;
        bdh->tsize = size;
#if BufStats
        thr->totalloc += (size_t)size;
        thr->numget++; /* Increment number of bget() calls */
        thr->numdget++; /* Direct bget() call count */
#endif
        buf = (void *)(bdh + 1);
        KMP_DEBUG_ASSERT(((size_t)buf) % SizeQuant == 0);
        return buf;
      }

    } else {

      /*  Try to obtain a new expansion block */
      void *newpool;

      KE_TRACE(10, ("%%%%%% MALLOCB( %d )\n", (int)thr->exp_incr));

      /* richryan */
      newpool = (*thr->acqfcn)((bufsize)thr->exp_incr);
      KMP_DEBUG_ASSERT(((size_t)newpool) % SizeQuant == 0);
      if (newpool != NULL) {
        bpool(th, newpool, thr->exp_incr);
        buf = bget(
            th, requested_size); /* This can't, I say, can't get into a loop. */
        return buf;
      }
    }
  }

  /*  Still no buffer available */

  return NULL;
}

/*  BGETZ  --  Allocate a buffer and clear its contents to zero.  We clear
               the  entire  contents  of  the buffer to zero, not just the
               region requested by the caller. */

static void *bgetz(kmp_info_t *th, bufsize size) {
  char *buf = (char *)bget(th, size);

  if (buf != NULL) {
    bhead_t *b;
    bufsize rsize;

    b = BH(buf - sizeof(bhead_t));
    rsize = -(b->bb.bsize);
    if (rsize == 0) {
      bdhead_t *bd;

      bd = BDH(buf - sizeof(bdhead_t));
      rsize = bd->tsize - (bufsize)sizeof(bdhead_t);
    } else {
      rsize -= sizeof(bhead_t);
    }

    KMP_DEBUG_ASSERT(rsize >= size);

    (void)memset(buf, 0, (bufsize)rsize);
  }
  return ((void *)buf);
}

/*  BGETR  --  Reallocate a buffer.  This is a minimal implementation,
               simply in terms of brel()  and  bget().   It  could  be
               enhanced to allow the buffer to grow into adjacent free
               blocks and to avoid moving data unnecessarily.  */

static void *bgetr(kmp_info_t *th, void *buf, bufsize size) {
  void *nbuf;
  bufsize osize; /* Old size of buffer */
  bhead_t *b;

  nbuf = bget(th, size);
  if (nbuf == NULL) { /* Acquire new buffer */
    return NULL;
  }
  if (buf == NULL) {
    return nbuf;
  }
  b = BH(((char *)buf) - sizeof(bhead_t));
  osize = -b->bb.bsize;
  if (osize == 0) {
    /*  Buffer acquired directly through acqfcn. */
    bdhead_t *bd;

    bd = BDH(((char *)buf) - sizeof(bdhead_t));
    osize = bd->tsize - (bufsize)sizeof(bdhead_t);
  } else {
    osize -= sizeof(bhead_t);
  }

  KMP_DEBUG_ASSERT(osize > 0);

  (void)KMP_MEMCPY((char *)nbuf, (char *)buf, /* Copy the data */
                   (size_t)((size < osize) ? size : osize));
  brel(th, buf);

  return nbuf;
}

/*  BREL  --  Release a buffer.  */
static void brel(kmp_info_t *th, void *buf) {
  thr_data_t *thr = get_thr_data(th);
  bfhead_t *b, *bn;
  kmp_info_t *bth;

  KMP_DEBUG_ASSERT(buf != NULL);
  KMP_DEBUG_ASSERT(((size_t)buf) % SizeQuant == 0);

  b = BFH(((char *)buf) - sizeof(bhead_t));

  if (b->bh.bb.bsize == 0) { /* Directly-acquired buffer? */
    bdhead_t *bdh;

    bdh = BDH(((char *)buf) - sizeof(bdhead_t));
    KMP_DEBUG_ASSERT(b->bh.bb.prevfree == 0);
#if BufStats
    thr->totalloc -= (size_t)bdh->tsize;
    thr->numdrel++; /* Number of direct releases */
    thr->numrel++; /* Increment number of brel() calls */
#endif /* BufStats */
#ifdef FreeWipe
    (void)memset((char *)buf, 0x55, (size_t)(bdh->tsize - sizeof(bdhead_t)));
#endif /* FreeWipe */

    KE_TRACE(10, ("%%%%%% FREE( %p )\n", (void *)bdh));

    KMP_DEBUG_ASSERT(thr->relfcn != 0);
    (*thr->relfcn)((void *)bdh); /* Release it directly. */
    return;
  }

  bth = (kmp_info_t *)((kmp_uintptr_t)TCR_PTR(b->bh.bb.bthr) &
                       ~1); // clear possible mark before comparison
  if (bth != th) {
    /* Add this buffer to be released by the owning thread later */
    __kmp_bget_enqueue(bth, buf
#ifdef USE_QUEUING_LOCK_FOR_BGET
                       ,
                       __kmp_gtid_from_thread(th)
#endif
    );
    return;
  }

  /* Buffer size must be negative, indicating that the buffer is allocated. */
  if (b->bh.bb.bsize >= 0) {
    bn = NULL;
  }
  KMP_DEBUG_ASSERT(b->bh.bb.bsize < 0);

  /*  Back pointer in next buffer must be zero, indicating the same thing: */

  KMP_DEBUG_ASSERT(BH((char *)b - b->bh.bb.bsize)->bb.prevfree == 0);

#if BufStats
  thr->numrel++; /* Increment number of brel() calls */
  thr->totalloc += (size_t)b->bh.bb.bsize;
#endif

  /* If the back link is nonzero, the previous buffer is free.  */

  if (b->bh.bb.prevfree != 0) {
    /* The previous buffer is free. Consolidate this buffer with it by adding
       the length of this buffer to the previous free buffer. Note that we
       subtract the size in the buffer being released, since it's negative to
       indicate that the buffer is allocated. */
    bufsize size = b->bh.bb.bsize;

    /* Make the previous buffer the one we're working on. */
    KMP_DEBUG_ASSERT(BH((char *)b - b->bh.bb.prevfree)->bb.bsize ==
                     b->bh.bb.prevfree);
    b = BFH(((char *)b) - b->bh.bb.prevfree);
    b->bh.bb.bsize -= size;

    /* unlink the buffer from the old freelist */
    __kmp_bget_remove_from_freelist(b);
  } else {
    /* The previous buffer isn't allocated. Mark this buffer size as positive
       (i.e. free) and fall through to place the buffer on the free list as an
       isolated free block. */
    b->bh.bb.bsize = -b->bh.bb.bsize;
  }

  /* insert buffer back onto a new freelist */
  __kmp_bget_insert_into_freelist(thr, b);

  /* Now we look at the next buffer in memory, located by advancing from
     the  start  of  this  buffer  by its size, to see if that buffer is
     free.  If it is, we combine  this  buffer  with  the  next  one  in
     memory, dechaining the second buffer from the free list. */
  bn = BFH(((char *)b) + b->bh.bb.bsize);
  if (bn->bh.bb.bsize > 0) {

    /* The buffer is free.  Remove it from the free list and add
       its size to that of our buffer. */
    KMP_DEBUG_ASSERT(BH((char *)bn + bn->bh.bb.bsize)->bb.prevfree ==
                     bn->bh.bb.bsize);

    __kmp_bget_remove_from_freelist(bn);

    b->bh.bb.bsize += bn->bh.bb.bsize;

    /* unlink the buffer from the old freelist, and reinsert it into the new
     * freelist */
    __kmp_bget_remove_from_freelist(b);
    __kmp_bget_insert_into_freelist(thr, b);

    /* Finally,  advance  to   the  buffer  that   follows  the  newly
       consolidated free block.  We must set its  backpointer  to  the
       head  of  the  consolidated free block.  We know the next block
       must be an allocated block because the process of recombination
       guarantees  that  two  free  blocks will never be contiguous in
       memory.  */
    bn = BFH(((char *)b) + b->bh.bb.bsize);
  }
#ifdef FreeWipe
  (void)memset(((char *)b) + sizeof(bfhead_t), 0x55,
               (size_t)(b->bh.bb.bsize - sizeof(bfhead_t)));
#endif
  KMP_DEBUG_ASSERT(bn->bh.bb.bsize < 0);

  /* The next buffer is allocated.  Set the backpointer in it  to  point
     to this buffer; the previous free buffer in memory. */

  bn->bh.bb.prevfree = b->bh.bb.bsize;

  /*  If  a  block-release function is defined, and this free buffer
      constitutes the entire block, release it.  Note that  pool_len
      is  defined  in  such a way that the test will fail unless all
      pool blocks are the same size.  */
  if (thr->relfcn != 0 &&
      b->bh.bb.bsize == (bufsize)(thr->pool_len - sizeof(bhead_t))) {
#if BufStats
    if (thr->numpblk !=
        1) { /* Do not release the last buffer until finalization time */
#endif

      KMP_DEBUG_ASSERT(b->bh.bb.prevfree == 0);
      KMP_DEBUG_ASSERT(BH((char *)b + b->bh.bb.bsize)->bb.bsize == ESent);
      KMP_DEBUG_ASSERT(BH((char *)b + b->bh.bb.bsize)->bb.prevfree ==
                       b->bh.bb.bsize);

      /*  Unlink the buffer from the free list  */
      __kmp_bget_remove_from_freelist(b);

      KE_TRACE(10, ("%%%%%% FREE( %p )\n", (void *)b));

      (*thr->relfcn)(b);
#if BufStats
      thr->numprel++; /* Nr of expansion block releases */
      thr->numpblk--; /* Total number of blocks */
      KMP_DEBUG_ASSERT(thr->numpblk == thr->numpget - thr->numprel);

      // avoid leaving stale last_pool pointer around if it is being dealloced
      if (thr->last_pool == b)
        thr->last_pool = 0;
    } else {
      thr->last_pool = b;
    }
#endif /* BufStats */
  }
}

/*  BECTL  --  Establish automatic pool expansion control  */
static void bectl(kmp_info_t *th, bget_compact_t compact,
                  bget_acquire_t acquire, bget_release_t release,
                  bufsize pool_incr) {
  thr_data_t *thr = get_thr_data(th);

  thr->compfcn = compact;
  thr->acqfcn = acquire;
  thr->relfcn = release;
  thr->exp_incr = pool_incr;
}

/*  BPOOL  --  Add a region of memory to the buffer pool.  */
static void bpool(kmp_info_t *th, void *buf, bufsize len) {
  /*    int bin = 0; */
  thr_data_t *thr = get_thr_data(th);
  bfhead_t *b = BFH(buf);
  bhead_t *bn;

  __kmp_bget_dequeue(th); /* Release any queued buffers */

#ifdef SizeQuant
  len &= ~((bufsize)(SizeQuant - 1));
#endif
  if (thr->pool_len == 0) {
    thr->pool_len = len;
  } else if (len != thr->pool_len) {
    thr->pool_len = -1;
  }
#if BufStats
  thr->numpget++; /* Number of block acquisitions */
  thr->numpblk++; /* Number of blocks total */
  KMP_DEBUG_ASSERT(thr->numpblk == thr->numpget - thr->numprel);
#endif /* BufStats */

  /* Since the block is initially occupied by a single free  buffer,
     it  had  better  not  be  (much) larger than the largest buffer
     whose size we can store in bhead.bb.bsize. */
  KMP_DEBUG_ASSERT(len - sizeof(bhead_t) <= -((bufsize)ESent + 1));

  /* Clear  the  backpointer at  the start of the block to indicate that
     there  is  no  free  block  prior  to  this   one.    That   blocks
     recombination when the first block in memory is released. */
  b->bh.bb.prevfree = 0;

  /* Create a dummy allocated buffer at the end of the pool.  This dummy
     buffer is seen when a buffer at the end of the pool is released and
     blocks  recombination  of  the last buffer with the dummy buffer at
     the end.  The length in the dummy buffer  is  set  to  the  largest
     negative  number  to  denote  the  end  of  the pool for diagnostic
     routines (this specific value is  not  counted  on  by  the  actual
     allocation and release functions). */
  len -= sizeof(bhead_t);
  b->bh.bb.bsize = (bufsize)len;
  /* Set the owner of this buffer */
  TCW_PTR(b->bh.bb.bthr,
          (kmp_info_t *)((kmp_uintptr_t)th |
                         1)); // mark the buffer as allocated address

  /* Chain the new block to the free list. */
  __kmp_bget_insert_into_freelist(thr, b);

#ifdef FreeWipe
  (void)memset(((char *)b) + sizeof(bfhead_t), 0x55,
               (size_t)(len - sizeof(bfhead_t)));
#endif
  bn = BH(((char *)b) + len);
  bn->bb.prevfree = (bufsize)len;
  /* Definition of ESent assumes two's complement! */
  KMP_DEBUG_ASSERT((~0) == -1 && (bn != 0));

  bn->bb.bsize = ESent;
}

/*  BFREED  --  Dump the free lists for this thread. */
static void bfreed(kmp_info_t *th) {
  int bin = 0, count = 0;
  int gtid = __kmp_gtid_from_thread(th);
  thr_data_t *thr = get_thr_data(th);

#if BufStats
  __kmp_printf_no_lock("__kmp_printpool: T#%d total=%" KMP_UINT64_SPEC
                       " get=%" KMP_INT64_SPEC " rel=%" KMP_INT64_SPEC
                       " pblk=%" KMP_INT64_SPEC " pget=%" KMP_INT64_SPEC
                       " prel=%" KMP_INT64_SPEC " dget=%" KMP_INT64_SPEC
                       " drel=%" KMP_INT64_SPEC "\n",
                       gtid, (kmp_uint64)thr->totalloc, (kmp_int64)thr->numget,
                       (kmp_int64)thr->numrel, (kmp_int64)thr->numpblk,
                       (kmp_int64)thr->numpget, (kmp_int64)thr->numprel,
                       (kmp_int64)thr->numdget, (kmp_int64)thr->numdrel);
#endif

  for (bin = 0; bin < MAX_BGET_BINS; ++bin) {
    bfhead_t *b;

    for (b = thr->freelist[bin].ql.flink; b != &thr->freelist[bin];
         b = b->ql.flink) {
      bufsize bs = b->bh.bb.bsize;

      KMP_DEBUG_ASSERT(b->ql.blink->ql.flink == b);
      KMP_DEBUG_ASSERT(b->ql.flink->ql.blink == b);
      KMP_DEBUG_ASSERT(bs > 0);

      count += 1;

      __kmp_printf_no_lock(
          "__kmp_printpool: T#%d Free block: 0x%p size %6ld bytes.\n", gtid, b,
          (long)bs);
#ifdef FreeWipe
      {
        char *lerr = ((char *)b) + sizeof(bfhead_t);
        if ((bs > sizeof(bfhead_t)) &&
            ((*lerr != 0x55) ||
             (memcmp(lerr, lerr + 1, (size_t)(bs - (sizeof(bfhead_t) + 1))) !=
              0))) {
          __kmp_printf_no_lock("__kmp_printpool: T#%d     (Contents of above "
                               "free block have been overstored.)\n",
                               gtid);
        }
      }
#endif
    }
  }

  if (count == 0)
    __kmp_printf_no_lock("__kmp_printpool: T#%d No free blocks\n", gtid);
}

void __kmp_initialize_bget(kmp_info_t *th) {
  KMP_DEBUG_ASSERT(SizeQuant >= sizeof(void *) && (th != 0));

  set_thr_data(th);

  bectl(th, (bget_compact_t)0, (bget_acquire_t)malloc, (bget_release_t)free,
        (bufsize)__kmp_malloc_pool_incr);
}

void __kmp_finalize_bget(kmp_info_t *th) {
  thr_data_t *thr;
  bfhead_t *b;

  KMP_DEBUG_ASSERT(th != 0);

#if BufStats
  thr = (thr_data_t *)th->th.th_local.bget_data;
  KMP_DEBUG_ASSERT(thr != NULL);
  b = thr->last_pool;

  /*  If a block-release function is defined, and this free buffer constitutes
      the entire block, release it. Note that pool_len is defined in such a way
      that the test will fail unless all pool blocks are the same size.  */

  // Deallocate the last pool if one exists because we no longer do it in brel()
  if (thr->relfcn != 0 && b != 0 && thr->numpblk != 0 &&
      b->bh.bb.bsize == (bufsize)(thr->pool_len - sizeof(bhead_t))) {
    KMP_DEBUG_ASSERT(b->bh.bb.prevfree == 0);
    KMP_DEBUG_ASSERT(BH((char *)b + b->bh.bb.bsize)->bb.bsize == ESent);
    KMP_DEBUG_ASSERT(BH((char *)b + b->bh.bb.bsize)->bb.prevfree ==
                     b->bh.bb.bsize);

    /*  Unlink the buffer from the free list  */
    __kmp_bget_remove_from_freelist(b);

    KE_TRACE(10, ("%%%%%% FREE( %p )\n", (void *)b));

    (*thr->relfcn)(b);
    thr->numprel++; /* Nr of expansion block releases */
    thr->numpblk--; /* Total number of blocks */
    KMP_DEBUG_ASSERT(thr->numpblk == thr->numpget - thr->numprel);
  }
#endif /* BufStats */

  /* Deallocate bget_data */
  if (th->th.th_local.bget_data != NULL) {
    __kmp_free(th->th.th_local.bget_data);
    th->th.th_local.bget_data = NULL;
  }
}

void kmpc_set_poolsize(size_t size) {
  bectl(__kmp_get_thread(), (bget_compact_t)0, (bget_acquire_t)malloc,
        (bget_release_t)free, (bufsize)size);
}

size_t kmpc_get_poolsize(void) {
  thr_data_t *p;

  p = get_thr_data(__kmp_get_thread());

  return p->exp_incr;
}

void kmpc_set_poolmode(int mode) {
  thr_data_t *p;

  if (mode == bget_mode_fifo || mode == bget_mode_lifo ||
      mode == bget_mode_best) {
    p = get_thr_data(__kmp_get_thread());
    p->mode = (bget_mode_t)mode;
  }
}

int kmpc_get_poolmode(void) {
  thr_data_t *p;

  p = get_thr_data(__kmp_get_thread());

  return p->mode;
}

void kmpc_get_poolstat(size_t *maxmem, size_t *allmem) {
  kmp_info_t *th = __kmp_get_thread();
  bufsize a, b;

  __kmp_bget_dequeue(th); /* Release any queued buffers */

  bcheck(th, &a, &b);

  *maxmem = a;
  *allmem = b;
}

void kmpc_poolprint(void) {
  kmp_info_t *th = __kmp_get_thread();

  __kmp_bget_dequeue(th); /* Release any queued buffers */

  bfreed(th);
}

#endif // #if KMP_USE_BGET

void *kmpc_malloc(size_t size) {
  void *ptr;
  ptr = bget(__kmp_entry_thread(), (bufsize)(size + sizeof(ptr)));
  if (ptr != NULL) {
    // save allocated pointer just before one returned to user
    *(void **)ptr = ptr;
    ptr = (void **)ptr + 1;
  }
  return ptr;
}

#define IS_POWER_OF_TWO(n) (((n) & ((n)-1)) == 0)

void *kmpc_aligned_malloc(size_t size, size_t alignment) {
  void *ptr;
  void *ptr_allocated;
  KMP_DEBUG_ASSERT(alignment < 32 * 1024); // Alignment should not be too big
  if (!IS_POWER_OF_TWO(alignment)) {
    // AC: do we need to issue a warning here?
    errno = EINVAL;
    return NULL;
  }
  size = size + sizeof(void *) + alignment;
  ptr_allocated = bget(__kmp_entry_thread(), (bufsize)size);
  if (ptr_allocated != NULL) {
    // save allocated pointer just before one returned to user
    ptr = (void *)(((kmp_uintptr_t)ptr_allocated + sizeof(void *) + alignment) &
                   ~(alignment - 1));
    *((void **)ptr - 1) = ptr_allocated;
  } else {
    ptr = NULL;
  }
  return ptr;
}

void *kmpc_calloc(size_t nelem, size_t elsize) {
  void *ptr;
  ptr = bgetz(__kmp_entry_thread(), (bufsize)(nelem * elsize + sizeof(ptr)));
  if (ptr != NULL) {
    // save allocated pointer just before one returned to user
    *(void **)ptr = ptr;
    ptr = (void **)ptr + 1;
  }
  return ptr;
}

void *kmpc_realloc(void *ptr, size_t size) {
  void *result = NULL;
  if (ptr == NULL) {
    // If pointer is NULL, realloc behaves like malloc.
    result = bget(__kmp_entry_thread(), (bufsize)(size + sizeof(ptr)));
    // save allocated pointer just before one returned to user
    if (result != NULL) {
      *(void **)result = result;
      result = (void **)result + 1;
    }
  } else if (size == 0) {
    // If size is 0, realloc behaves like free.
    // The thread must be registered by the call to kmpc_malloc() or
    // kmpc_calloc() before.
    // So it should be safe to call __kmp_get_thread(), not
    // __kmp_entry_thread().
    KMP_ASSERT(*((void **)ptr - 1));
    brel(__kmp_get_thread(), *((void **)ptr - 1));
  } else {
    result = bgetr(__kmp_entry_thread(), *((void **)ptr - 1),
                   (bufsize)(size + sizeof(ptr)));
    if (result != NULL) {
      *(void **)result = result;
      result = (void **)result + 1;
    }
  }
  return result;
}

// NOTE: the library must have already been initialized by a previous allocate
void kmpc_free(void *ptr) {
  if (!__kmp_init_serial) {
    return;
  }
  if (ptr != NULL) {
    kmp_info_t *th = __kmp_get_thread();
    __kmp_bget_dequeue(th); /* Release any queued buffers */
    // extract allocated pointer and free it
    KMP_ASSERT(*((void **)ptr - 1));
    brel(th, *((void **)ptr - 1));
  }
}

void *___kmp_thread_malloc(kmp_info_t *th, size_t size KMP_SRC_LOC_DECL) {
  void *ptr;
  KE_TRACE(30, ("-> __kmp_thread_malloc( %p, %d ) called from %s:%d\n", th,
                (int)size KMP_SRC_LOC_PARM));
  ptr = bget(th, (bufsize)size);
  KE_TRACE(30, ("<- __kmp_thread_malloc() returns %p\n", ptr));
  return ptr;
}

void *___kmp_thread_calloc(kmp_info_t *th, size_t nelem,
                           size_t elsize KMP_SRC_LOC_DECL) {
  void *ptr;
  KE_TRACE(30, ("-> __kmp_thread_calloc( %p, %d, %d ) called from %s:%d\n", th,
                (int)nelem, (int)elsize KMP_SRC_LOC_PARM));
  ptr = bgetz(th, (bufsize)(nelem * elsize));
  KE_TRACE(30, ("<- __kmp_thread_calloc() returns %p\n", ptr));
  return ptr;
}

void *___kmp_thread_realloc(kmp_info_t *th, void *ptr,
                            size_t size KMP_SRC_LOC_DECL) {
  KE_TRACE(30, ("-> __kmp_thread_realloc( %p, %p, %d ) called from %s:%d\n", th,
                ptr, (int)size KMP_SRC_LOC_PARM));
  ptr = bgetr(th, ptr, (bufsize)size);
  KE_TRACE(30, ("<- __kmp_thread_realloc() returns %p\n", ptr));
  return ptr;
}

void ___kmp_thread_free(kmp_info_t *th, void *ptr KMP_SRC_LOC_DECL) {
  KE_TRACE(30, ("-> __kmp_thread_free( %p, %p ) called from %s:%d\n", th,
                ptr KMP_SRC_LOC_PARM));
  if (ptr != NULL) {
    __kmp_bget_dequeue(th); /* Release any queued buffers */
    brel(th, ptr);
  }
  KE_TRACE(30, ("<- __kmp_thread_free()\n"));
}

/* OMP 5.0 Memory Management support */
static const char *kmp_mk_lib_name;
static void *h_memkind;
/* memkind experimental API: */
// memkind_alloc
static void *(*kmp_mk_alloc)(void *k, size_t sz);
// memkind_free
static void (*kmp_mk_free)(void *kind, void *ptr);
// memkind_check_available
static int (*kmp_mk_check)(void *kind);
// kinds we are going to use
static void **mk_default;
static void **mk_interleave;
static void **mk_hbw;
static void **mk_hbw_interleave;
static void **mk_hbw_preferred;
static void **mk_hugetlb;
static void **mk_hbw_hugetlb;
static void **mk_hbw_preferred_hugetlb;
static void **mk_dax_kmem;
static void **mk_dax_kmem_all;
static void **mk_dax_kmem_preferred;
static void *(*kmp_target_alloc_host)(size_t size, int device);
static void *(*kmp_target_alloc_shared)(size_t size, int device);
static void *(*kmp_target_alloc_device)(size_t size, int device);
static void *(*kmp_target_free_host)(void *ptr, int device);
static void *(*kmp_target_free_shared)(void *ptr, int device);
static void *(*kmp_target_free_device)(void *ptr, int device);
static bool __kmp_target_mem_available;
#define KMP_IS_TARGET_MEM_SPACE(MS)                                            \
  (MS == llvm_omp_target_host_mem_space ||                                     \
   MS == llvm_omp_target_shared_mem_space ||                                   \
   MS == llvm_omp_target_device_mem_space)
#define KMP_IS_TARGET_MEM_ALLOC(MA)                                            \
  (MA == llvm_omp_target_host_mem_alloc ||                                     \
   MA == llvm_omp_target_shared_mem_alloc ||                                   \
   MA == llvm_omp_target_device_mem_alloc)

#if KMP_OS_UNIX && KMP_DYNAMIC_LIB && !KMP_OS_DARWIN
static inline void chk_kind(void ***pkind) {
  KMP_DEBUG_ASSERT(pkind);
  if (*pkind) // symbol found
    if (kmp_mk_check(**pkind)) // kind not available or error
      *pkind = NULL;
}
#endif

void __kmp_init_memkind() {
// as of 2018-07-31 memkind does not support Windows*, exclude it for now
#if KMP_OS_UNIX && KMP_DYNAMIC_LIB && !KMP_OS_DARWIN
  // use of statically linked memkind is problematic, as it depends on libnuma
  kmp_mk_lib_name = "libmemkind.so";
  h_memkind = dlopen(kmp_mk_lib_name, RTLD_LAZY);
  if (h_memkind) {
    kmp_mk_check = (int (*)(void *))dlsym(h_memkind, "memkind_check_available");
    kmp_mk_alloc =
        (void *(*)(void *, size_t))dlsym(h_memkind, "memkind_malloc");
    kmp_mk_free = (void (*)(void *, void *))dlsym(h_memkind, "memkind_free");
    mk_default = (void **)dlsym(h_memkind, "MEMKIND_DEFAULT");
    if (kmp_mk_check && kmp_mk_alloc && kmp_mk_free && mk_default &&
        !kmp_mk_check(*mk_default)) {
      __kmp_memkind_available = 1;
      mk_interleave = (void **)dlsym(h_memkind, "MEMKIND_INTERLEAVE");
      chk_kind(&mk_interleave);
      mk_hbw = (void **)dlsym(h_memkind, "MEMKIND_HBW");
      chk_kind(&mk_hbw);
      mk_hbw_interleave = (void **)dlsym(h_memkind, "MEMKIND_HBW_INTERLEAVE");
      chk_kind(&mk_hbw_interleave);
      mk_hbw_preferred = (void **)dlsym(h_memkind, "MEMKIND_HBW_PREFERRED");
      chk_kind(&mk_hbw_preferred);
      mk_hugetlb = (void **)dlsym(h_memkind, "MEMKIND_HUGETLB");
      chk_kind(&mk_hugetlb);
      mk_hbw_hugetlb = (void **)dlsym(h_memkind, "MEMKIND_HBW_HUGETLB");
      chk_kind(&mk_hbw_hugetlb);
      mk_hbw_preferred_hugetlb =
          (void **)dlsym(h_memkind, "MEMKIND_HBW_PREFERRED_HUGETLB");
      chk_kind(&mk_hbw_preferred_hugetlb);
      mk_dax_kmem = (void **)dlsym(h_memkind, "MEMKIND_DAX_KMEM");
      chk_kind(&mk_dax_kmem);
      mk_dax_kmem_all = (void **)dlsym(h_memkind, "MEMKIND_DAX_KMEM_ALL");
      chk_kind(&mk_dax_kmem_all);
      mk_dax_kmem_preferred =
          (void **)dlsym(h_memkind, "MEMKIND_DAX_KMEM_PREFERRED");
      chk_kind(&mk_dax_kmem_preferred);
      KE_TRACE(25, ("__kmp_init_memkind: memkind library initialized\n"));
      return; // success
    }
    dlclose(h_memkind); // failure
  }
#else // !(KMP_OS_UNIX && KMP_DYNAMIC_LIB)
  kmp_mk_lib_name = "";
#endif // !(KMP_OS_UNIX && KMP_DYNAMIC_LIB)
  h_memkind = NULL;
  kmp_mk_check = NULL;
  kmp_mk_alloc = NULL;
  kmp_mk_free = NULL;
  mk_default = NULL;
  mk_interleave = NULL;
  mk_hbw = NULL;
  mk_hbw_interleave = NULL;
  mk_hbw_preferred = NULL;
  mk_hugetlb = NULL;
  mk_hbw_hugetlb = NULL;
  mk_hbw_preferred_hugetlb = NULL;
  mk_dax_kmem = NULL;
  mk_dax_kmem_all = NULL;
  mk_dax_kmem_preferred = NULL;
}

void __kmp_fini_memkind() {
#if KMP_OS_UNIX && KMP_DYNAMIC_LIB
  if (__kmp_memkind_available)
    KE_TRACE(25, ("__kmp_fini_memkind: finalize memkind library\n"));
  if (h_memkind) {
    dlclose(h_memkind);
    h_memkind = NULL;
  }
  kmp_mk_check = NULL;
  kmp_mk_alloc = NULL;
  kmp_mk_free = NULL;
  mk_default = NULL;
  mk_interleave = NULL;
  mk_hbw = NULL;
  mk_hbw_interleave = NULL;
  mk_hbw_preferred = NULL;
  mk_hugetlb = NULL;
  mk_hbw_hugetlb = NULL;
  mk_hbw_preferred_hugetlb = NULL;
  mk_dax_kmem = NULL;
  mk_dax_kmem_all = NULL;
  mk_dax_kmem_preferred = NULL;
#endif
}

void __kmp_init_target_mem() {
  *(void **)(&kmp_target_alloc_host) = KMP_DLSYM("llvm_omp_target_alloc_host");
  *(void **)(&kmp_target_alloc_shared) =
      KMP_DLSYM("llvm_omp_target_alloc_shared");
  *(void **)(&kmp_target_alloc_device) =
      KMP_DLSYM("llvm_omp_target_alloc_device");
  *(void **)(&kmp_target_free_host) = KMP_DLSYM("llvm_omp_target_free_host");
  *(void **)(&kmp_target_free_shared) =
      KMP_DLSYM("llvm_omp_target_free_shared");
  *(void **)(&kmp_target_free_device) =
      KMP_DLSYM("llvm_omp_target_free_device");
  __kmp_target_mem_available =
      kmp_target_alloc_host && kmp_target_alloc_shared &&
      kmp_target_alloc_device && kmp_target_free_host &&
      kmp_target_free_shared && kmp_target_free_device;
}

omp_allocator_handle_t __kmpc_init_allocator(int gtid, omp_memspace_handle_t ms,
                                             int ntraits,
                                             omp_alloctrait_t traits[]) {
  // OpenMP 5.0 only allows predefined memspaces
  KMP_DEBUG_ASSERT(ms == omp_default_mem_space || ms == omp_low_lat_mem_space ||
                   ms == omp_large_cap_mem_space || ms == omp_const_mem_space ||
                   ms == omp_high_bw_mem_space || KMP_IS_TARGET_MEM_SPACE(ms));
  kmp_allocator_t *al;
  int i;
  al = (kmp_allocator_t *)__kmp_allocate(sizeof(kmp_allocator_t)); // zeroed
  al->memspace = ms; // not used currently
  for (i = 0; i < ntraits; ++i) {
    switch (traits[i].key) {
    case omp_atk_sync_hint:
    case omp_atk_access:
    case omp_atk_pinned:
      break;
    case omp_atk_alignment:
      __kmp_type_convert(traits[i].value, &(al->alignment));
      KMP_ASSERT(IS_POWER_OF_TWO(al->alignment));
      break;
    case omp_atk_pool_size:
      al->pool_size = traits[i].value;
      break;
    case omp_atk_fallback:
      al->fb = (omp_alloctrait_value_t)traits[i].value;
      KMP_DEBUG_ASSERT(
          al->fb == omp_atv_default_mem_fb || al->fb == omp_atv_null_fb ||
          al->fb == omp_atv_abort_fb || al->fb == omp_atv_allocator_fb);
      break;
    case omp_atk_fb_data:
      al->fb_data = RCAST(kmp_allocator_t *, traits[i].value);
      break;
    case omp_atk_partition:
      al->memkind = RCAST(void **, traits[i].value);
      break;
    default:
      KMP_ASSERT2(0, "Unexpected allocator trait");
    }
  }
  if (al->fb == 0) {
    // set default allocator
    al->fb = omp_atv_default_mem_fb;
    al->fb_data = (kmp_allocator_t *)omp_default_mem_alloc;
  } else if (al->fb == omp_atv_allocator_fb) {
    KMP_ASSERT(al->fb_data != NULL);
  } else if (al->fb == omp_atv_default_mem_fb) {
    al->fb_data = (kmp_allocator_t *)omp_default_mem_alloc;
  }
  if (__kmp_memkind_available) {
    // Let's use memkind library if available
    if (ms == omp_high_bw_mem_space) {
      if (al->memkind == (void *)omp_atv_interleaved && mk_hbw_interleave) {
        al->memkind = mk_hbw_interleave;
      } else if (mk_hbw_preferred) {
        // AC: do not try to use MEMKIND_HBW for now, because memkind library
        // cannot reliably detect exhaustion of HBW memory.
        // It could be possible using hbw_verify_memory_region() but memkind
        // manual says: "Using this function in production code may result in
        // serious performance penalty".
        al->memkind = mk_hbw_preferred;
      } else {
        // HBW is requested but not available --> return NULL allocator
        __kmp_free(al);
        return omp_null_allocator;
      }
    } else if (ms == omp_large_cap_mem_space) {
      if (mk_dax_kmem_all) {
        // All pmem nodes are visited
        al->memkind = mk_dax_kmem_all;
      } else if (mk_dax_kmem) {
        // Only closest pmem node is visited
        al->memkind = mk_dax_kmem;
      } else {
        __kmp_free(al);
        return omp_null_allocator;
      }
    } else {
      if (al->memkind == (void *)omp_atv_interleaved && mk_interleave) {
        al->memkind = mk_interleave;
      } else {
        al->memkind = mk_default;
      }
    }
  } else if (KMP_IS_TARGET_MEM_SPACE(ms) && !__kmp_target_mem_available) {
    __kmp_free(al);
    return omp_null_allocator;
  } else {
    if (ms == omp_high_bw_mem_space) {
      // cannot detect HBW memory presence without memkind library
      __kmp_free(al);
      return omp_null_allocator;
    }
  }
  return (omp_allocator_handle_t)al;
}

void __kmpc_destroy_allocator(int gtid, omp_allocator_handle_t allocator) {
  if (allocator > kmp_max_mem_alloc)
    __kmp_free(allocator);
}

void __kmpc_set_default_allocator(int gtid, omp_allocator_handle_t allocator) {
  if (allocator == omp_null_allocator)
    allocator = omp_default_mem_alloc;
  __kmp_threads[gtid]->th.th_def_allocator = allocator;
}

omp_allocator_handle_t __kmpc_get_default_allocator(int gtid) {
  return __kmp_threads[gtid]->th.th_def_allocator;
}

typedef struct kmp_mem_desc { // Memory block descriptor
  void *ptr_alloc; // Pointer returned by allocator
  size_t size_a; // Size of allocated memory block (initial+descriptor+align)
  size_t size_orig; // Original size requested
  void *ptr_align; // Pointer to aligned memory, returned
  kmp_allocator_t *allocator; // allocator
} kmp_mem_desc_t;
static int alignment = sizeof(void *); // align to pointer size by default

// external interfaces are wrappers over internal implementation
void *__kmpc_alloc(int gtid, size_t size, omp_allocator_handle_t allocator) {
  KE_TRACE(25, ("__kmpc_alloc: T#%d (%d, %p)\n", gtid, (int)size, allocator));
  void *ptr = __kmp_alloc(gtid, 0, size, allocator);
  KE_TRACE(25, ("__kmpc_alloc returns %p, T#%d\n", ptr, gtid));
  return ptr;
}

void *__kmpc_aligned_alloc(int gtid, size_t algn, size_t size,
                           omp_allocator_handle_t allocator) {
  KE_TRACE(25, ("__kmpc_aligned_alloc: T#%d (%d, %d, %p)\n", gtid, (int)algn,
                (int)size, allocator));
  void *ptr = __kmp_alloc(gtid, algn, size, allocator);
  KE_TRACE(25, ("__kmpc_aligned_alloc returns %p, T#%d\n", ptr, gtid));
  return ptr;
}

void *__kmpc_calloc(int gtid, size_t nmemb, size_t size,
                    omp_allocator_handle_t allocator) {
  KE_TRACE(25, ("__kmpc_calloc: T#%d (%d, %d, %p)\n", gtid, (int)nmemb,
                (int)size, allocator));
  void *ptr = __kmp_calloc(gtid, 0, nmemb, size, allocator);
  KE_TRACE(25, ("__kmpc_calloc returns %p, T#%d\n", ptr, gtid));
  return ptr;
}

void *__kmpc_realloc(int gtid, void *ptr, size_t size,
                     omp_allocator_handle_t allocator,
                     omp_allocator_handle_t free_allocator) {
  KE_TRACE(25, ("__kmpc_realloc: T#%d (%p, %d, %p, %p)\n", gtid, ptr, (int)size,
                allocator, free_allocator));
  void *nptr = __kmp_realloc(gtid, ptr, size, allocator, free_allocator);
  KE_TRACE(25, ("__kmpc_realloc returns %p, T#%d\n", nptr, gtid));
  return nptr;
}

void __kmpc_free(int gtid, void *ptr, omp_allocator_handle_t allocator) {
  KE_TRACE(25, ("__kmpc_free: T#%d free(%p,%p)\n", gtid, ptr, allocator));
  ___kmpc_free(gtid, ptr, allocator);
  KE_TRACE(10, ("__kmpc_free: T#%d freed %p (%p)\n", gtid, ptr, allocator));
  return;
}

// internal implementation, called from inside the library
void *__kmp_alloc(int gtid, size_t algn, size_t size,
                  omp_allocator_handle_t allocator) {
  void *ptr = NULL;
  kmp_allocator_t *al;
  KMP_DEBUG_ASSERT(__kmp_init_serial);
  if (size == 0)
    return NULL;
  if (allocator == omp_null_allocator)
    allocator = __kmp_threads[gtid]->th.th_def_allocator;

  al = RCAST(kmp_allocator_t *, allocator);

  int sz_desc = sizeof(kmp_mem_desc_t);
  kmp_mem_desc_t desc;
  kmp_uintptr_t addr; // address returned by allocator
  kmp_uintptr_t addr_align; // address to return to caller
  kmp_uintptr_t addr_descr; // address of memory block descriptor
  size_t align = alignment; // default alignment
  if (allocator > kmp_max_mem_alloc && al->alignment > align)
    align = al->alignment; // alignment required by allocator trait
  if (align < algn)
    align = algn; // max of allocator trait, parameter and sizeof(void*)
  desc.size_orig = size;
  desc.size_a = size + sz_desc + align;

  // Use default allocator if libmemkind is not available
  int use_default_allocator = (__kmp_memkind_available) ? false : true;

  if (__kmp_memkind_available) {
    if (allocator < kmp_max_mem_alloc) {
      // pre-defined allocator
      if (allocator == omp_high_bw_mem_alloc && mk_hbw_preferred) {
        ptr = kmp_mk_alloc(*mk_hbw_preferred, desc.size_a);
      } else if (allocator == omp_large_cap_mem_alloc && mk_dax_kmem_all) {
        ptr = kmp_mk_alloc(*mk_dax_kmem_all, desc.size_a);
      } else {
        ptr = kmp_mk_alloc(*mk_default, desc.size_a);
      }
    } else if (al->pool_size > 0) {
      // custom allocator with pool size requested
      kmp_uint64 used =
          KMP_TEST_THEN_ADD64((kmp_int64 *)&al->pool_used, desc.size_a);
      if (used + desc.size_a > al->pool_size) {
        // not enough space, need to go fallback path
        KMP_TEST_THEN_ADD64((kmp_int64 *)&al->pool_used, -desc.size_a);
        if (al->fb == omp_atv_default_mem_fb) {
          al = (kmp_allocator_t *)omp_default_mem_alloc;
          ptr = kmp_mk_alloc(*mk_default, desc.size_a);
        } else if (al->fb == omp_atv_abort_fb) {
          KMP_ASSERT(0); // abort fallback requested
        } else if (al->fb == omp_atv_allocator_fb) {
          KMP_ASSERT(al != al->fb_data);
          al = al->fb_data;
          return __kmp_alloc(gtid, algn, size, (omp_allocator_handle_t)al);
        } // else ptr == NULL;
      } else {
        // pool has enough space
        ptr = kmp_mk_alloc(*al->memkind, desc.size_a);
        if (ptr == NULL) {
          if (al->fb == omp_atv_default_mem_fb) {
            al = (kmp_allocator_t *)omp_default_mem_alloc;
            ptr = kmp_mk_alloc(*mk_default, desc.size_a);
          } else if (al->fb == omp_atv_abort_fb) {
            KMP_ASSERT(0); // abort fallback requested
          } else if (al->fb == omp_atv_allocator_fb) {
            KMP_ASSERT(al != al->fb_data);
            al = al->fb_data;
            return __kmp_alloc(gtid, algn, size, (omp_allocator_handle_t)al);
          }
        }
      }
    } else {
      // custom allocator, pool size not requested
      ptr = kmp_mk_alloc(*al->memkind, desc.size_a);
      if (ptr == NULL) {
        if (al->fb == omp_atv_default_mem_fb) {
          al = (kmp_allocator_t *)omp_default_mem_alloc;
          ptr = kmp_mk_alloc(*mk_default, desc.size_a);
        } else if (al->fb == omp_atv_abort_fb) {
          KMP_ASSERT(0); // abort fallback requested
        } else if (al->fb == omp_atv_allocator_fb) {
          KMP_ASSERT(al != al->fb_data);
          al = al->fb_data;
          return __kmp_alloc(gtid, algn, size, (omp_allocator_handle_t)al);
        }
      }
    }
  } else if (allocator < kmp_max_mem_alloc) {
    if (KMP_IS_TARGET_MEM_ALLOC(allocator)) {
      // Use size input directly as the memory may not be accessible on host.
      // Use default device for now.
      if (__kmp_target_mem_available) {
        kmp_int32 device =
            __kmp_threads[gtid]->th.th_current_task->td_icvs.default_device;
        if (allocator == llvm_omp_target_host_mem_alloc)
          ptr = kmp_target_alloc_host(size, device);
        else if (allocator == llvm_omp_target_shared_mem_alloc)
          ptr = kmp_target_alloc_shared(size, device);
        else // allocator == llvm_omp_target_device_mem_alloc
          ptr = kmp_target_alloc_device(size, device);
      }
      return ptr;
    }

    // pre-defined allocator
    if (allocator == omp_high_bw_mem_alloc) {
      KMP_WARNING(OmpNoAllocator, "omp_high_bw_mem_alloc");
    } else if (allocator == omp_large_cap_mem_alloc) {
      KMP_WARNING(OmpNoAllocator, "omp_large_cap_mem_alloc");
    } else if (allocator == omp_const_mem_alloc) {
      KMP_WARNING(OmpNoAllocator, "omp_const_mem_alloc");
    } else if (allocator == omp_low_lat_mem_alloc) {
      KMP_WARNING(OmpNoAllocator, "omp_low_lat_mem_alloc");
    } else if (allocator == omp_cgroup_mem_alloc) {
      KMP_WARNING(OmpNoAllocator, "omp_cgroup_mem_alloc");
    } else if (allocator == omp_pteam_mem_alloc) {
      KMP_WARNING(OmpNoAllocator, "omp_pteam_mem_alloc");
    } else if (allocator == omp_thread_mem_alloc) {
      KMP_WARNING(OmpNoAllocator, "omp_thread_mem_alloc");
    } else { // default allocator requested
      use_default_allocator = true;
    }
    if (use_default_allocator) {
      ptr = __kmp_thread_malloc(__kmp_thread_from_gtid(gtid), desc.size_a);
      use_default_allocator = false;
    }
  } else if (KMP_IS_TARGET_MEM_SPACE(al->memspace)) {
    if (__kmp_target_mem_available) {
      kmp_int32 device =
          __kmp_threads[gtid]->th.th_current_task->td_icvs.default_device;
      if (al->memspace == llvm_omp_target_host_mem_space)
        ptr = kmp_target_alloc_host(size, device);
      else if (al->memspace == llvm_omp_target_shared_mem_space)
        ptr = kmp_target_alloc_shared(size, device);
      else // al->memspace == llvm_omp_target_device_mem_space
        ptr = kmp_target_alloc_device(size, device);
    }
    return ptr;
  } else if (al->pool_size > 0) {
    // custom allocator with pool size requested
    kmp_uint64 used =
        KMP_TEST_THEN_ADD64((kmp_int64 *)&al->pool_used, desc.size_a);
    if (used + desc.size_a > al->pool_size) {
      // not enough space, need to go fallback path
      KMP_TEST_THEN_ADD64((kmp_int64 *)&al->pool_used, -desc.size_a);
      if (al->fb == omp_atv_default_mem_fb) {
        al = (kmp_allocator_t *)omp_default_mem_alloc;
        ptr = __kmp_thread_malloc(__kmp_thread_from_gtid(gtid), desc.size_a);
      } else if (al->fb == omp_atv_abort_fb) {
        KMP_ASSERT(0); // abort fallback requested
      } else if (al->fb == omp_atv_allocator_fb) {
        KMP_ASSERT(al != al->fb_data);
        al = al->fb_data;
        return __kmp_alloc(gtid, algn, size, (omp_allocator_handle_t)al);
      } // else ptr == NULL;
    } else {
      // pool has enough space
      ptr = __kmp_thread_malloc(__kmp_thread_from_gtid(gtid), desc.size_a);
      if (ptr == NULL && al->fb == omp_atv_abort_fb) {
        KMP_ASSERT(0); // abort fallback requested
      } // no sense to look for another fallback because of same internal alloc
    }
  } else {
    // custom allocator, pool size not requested
    ptr = __kmp_thread_malloc(__kmp_thread_from_gtid(gtid), desc.size_a);
    if (ptr == NULL && al->fb == omp_atv_abort_fb) {
      KMP_ASSERT(0); // abort fallback requested
    } // no sense to look for another fallback because of same internal alloc
  }
  KE_TRACE(10, ("__kmp_alloc: T#%d %p=alloc(%d)\n", gtid, ptr, desc.size_a));
  if (ptr == NULL)
    return NULL;

  addr = (kmp_uintptr_t)ptr;
  addr_align = (addr + sz_desc + align - 1) & ~(align - 1);
  addr_descr = addr_align - sz_desc;

  desc.ptr_alloc = ptr;
  desc.ptr_align = (void *)addr_align;
  desc.allocator = al;
  *((kmp_mem_desc_t *)addr_descr) = desc; // save descriptor contents
  KMP_MB();

  return desc.ptr_align;
}

void *__kmp_calloc(int gtid, size_t algn, size_t nmemb, size_t size,
                   omp_allocator_handle_t allocator) {
  void *ptr = NULL;
  kmp_allocator_t *al;
  KMP_DEBUG_ASSERT(__kmp_init_serial);

  if (allocator == omp_null_allocator)
    allocator = __kmp_threads[gtid]->th.th_def_allocator;

  al = RCAST(kmp_allocator_t *, allocator);

  if (nmemb == 0 || size == 0)
    return ptr;

  if ((SIZE_MAX - sizeof(kmp_mem_desc_t)) / size < nmemb) {
    if (al->fb == omp_atv_abort_fb) {
      KMP_ASSERT(0);
    }
    return ptr;
  }

  ptr = __kmp_alloc(gtid, algn, nmemb * size, allocator);

  if (ptr) {
    memset(ptr, 0x00, nmemb * size);
  }
  return ptr;
}

void *__kmp_realloc(int gtid, void *ptr, size_t size,
                    omp_allocator_handle_t allocator,
                    omp_allocator_handle_t free_allocator) {
  void *nptr = NULL;
  KMP_DEBUG_ASSERT(__kmp_init_serial);

  if (size == 0) {
    if (ptr != NULL)
      ___kmpc_free(gtid, ptr, free_allocator);
    return nptr;
  }

  nptr = __kmp_alloc(gtid, 0, size, allocator);

  if (nptr != NULL && ptr != NULL) {
    kmp_mem_desc_t desc;
    kmp_uintptr_t addr_align; // address to return to caller
    kmp_uintptr_t addr_descr; // address of memory block descriptor

    addr_align = (kmp_uintptr_t)ptr;
    addr_descr = addr_align - sizeof(kmp_mem_desc_t);
    desc = *((kmp_mem_desc_t *)addr_descr); // read descriptor

    KMP_DEBUG_ASSERT(desc.ptr_align == ptr);
    KMP_DEBUG_ASSERT(desc.size_orig > 0);
    KMP_DEBUG_ASSERT(desc.size_orig < desc.size_a);
    KMP_MEMCPY((char *)nptr, (char *)ptr,
               (size_t)((size < desc.size_orig) ? size : desc.size_orig));
  }

  if (nptr != NULL) {
    ___kmpc_free(gtid, ptr, free_allocator);
  }

  return nptr;
}

void ___kmpc_free(int gtid, void *ptr, omp_allocator_handle_t allocator) {
  if (ptr == NULL)
    return;

  kmp_allocator_t *al;
  omp_allocator_handle_t oal;
  al = RCAST(kmp_allocator_t *, CCAST(omp_allocator_handle_t, allocator));
  kmp_mem_desc_t desc;
  kmp_uintptr_t addr_align; // address to return to caller
  kmp_uintptr_t addr_descr; // address of memory block descriptor
  if (__kmp_target_mem_available && (KMP_IS_TARGET_MEM_ALLOC(allocator) ||
                                     (allocator > kmp_max_mem_alloc &&
                                      KMP_IS_TARGET_MEM_SPACE(al->memspace)))) {
    kmp_int32 device =
        __kmp_threads[gtid]->th.th_current_task->td_icvs.default_device;
    if (allocator == llvm_omp_target_host_mem_alloc) {
      kmp_target_free_host(ptr, device);
    } else if (allocator == llvm_omp_target_shared_mem_alloc) {
      kmp_target_free_shared(ptr, device);
    } else if (allocator == llvm_omp_target_device_mem_alloc) {
      kmp_target_free_device(ptr, device);
    }
    return;
  }

  addr_align = (kmp_uintptr_t)ptr;
  addr_descr = addr_align - sizeof(kmp_mem_desc_t);
  desc = *((kmp_mem_desc_t *)addr_descr); // read descriptor

  KMP_DEBUG_ASSERT(desc.ptr_align == ptr);
  if (allocator) {
    KMP_DEBUG_ASSERT(desc.allocator == al || desc.allocator == al->fb_data);
  }
  al = desc.allocator;
  oal = (omp_allocator_handle_t)al; // cast to void* for comparisons
  KMP_DEBUG_ASSERT(al);

  if (__kmp_memkind_available) {
    if (oal < kmp_max_mem_alloc) {
      // pre-defined allocator
      if (oal == omp_high_bw_mem_alloc && mk_hbw_preferred) {
        kmp_mk_free(*mk_hbw_preferred, desc.ptr_alloc);
      } else if (oal == omp_large_cap_mem_alloc && mk_dax_kmem_all) {
        kmp_mk_free(*mk_dax_kmem_all, desc.ptr_alloc);
      } else {
        kmp_mk_free(*mk_default, desc.ptr_alloc);
      }
    } else {
      if (al->pool_size > 0) { // custom allocator with pool size requested
        kmp_uint64 used =
            KMP_TEST_THEN_ADD64((kmp_int64 *)&al->pool_used, -desc.size_a);
        (void)used; // to suppress compiler warning
        KMP_DEBUG_ASSERT(used >= desc.size_a);
      }
      kmp_mk_free(*al->memkind, desc.ptr_alloc);
    }
  } else {
    if (oal > kmp_max_mem_alloc && al->pool_size > 0) {
      kmp_uint64 used =
          KMP_TEST_THEN_ADD64((kmp_int64 *)&al->pool_used, -desc.size_a);
      (void)used; // to suppress compiler warning
      KMP_DEBUG_ASSERT(used >= desc.size_a);
    }
    __kmp_thread_free(__kmp_thread_from_gtid(gtid), desc.ptr_alloc);
  }
}

/* If LEAK_MEMORY is defined, __kmp_free() will *not* free memory. It causes
   memory leaks, but it may be useful for debugging memory corruptions, used
   freed pointers, etc. */
/* #define LEAK_MEMORY */
struct kmp_mem_descr { // Memory block descriptor.
  void *ptr_allocated; // Pointer returned by malloc(), subject for free().
  size_t size_allocated; // Size of allocated memory block.
  void *ptr_aligned; // Pointer to aligned memory, to be used by client code.
  size_t size_aligned; // Size of aligned memory block.
};
typedef struct kmp_mem_descr kmp_mem_descr_t;

/* Allocate memory on requested boundary, fill allocated memory with 0x00.
   NULL is NEVER returned, __kmp_abort() is called in case of memory allocation
   error. Must use __kmp_free when freeing memory allocated by this routine! */
static void *___kmp_allocate_align(size_t size,
                                   size_t alignment KMP_SRC_LOC_DECL) {
  /* __kmp_allocate() allocates (by call to malloc()) bigger memory block than
     requested to return properly aligned pointer. Original pointer returned
     by malloc() and size of allocated block is saved in descriptor just
     before the aligned pointer. This information used by __kmp_free() -- it
     has to pass to free() original pointer, not aligned one.

          +---------+------------+-----------------------------------+---------+
          | padding | descriptor |           aligned block           | padding |
          +---------+------------+-----------------------------------+---------+
          ^                      ^
          |                      |
          |                      +- Aligned pointer returned to caller
          +- Pointer returned by malloc()

      Aligned block is filled with zeros, paddings are filled with 0xEF. */

  kmp_mem_descr_t descr;
  kmp_uintptr_t addr_allocated; // Address returned by malloc().
  kmp_uintptr_t addr_aligned; // Aligned address to return to caller.
  kmp_uintptr_t addr_descr; // Address of memory block descriptor.

  KE_TRACE(25, ("-> ___kmp_allocate_align( %d, %d ) called from %s:%d\n",
                (int)size, (int)alignment KMP_SRC_LOC_PARM));

  KMP_DEBUG_ASSERT(alignment < 32 * 1024); // Alignment should not be too
  KMP_DEBUG_ASSERT(sizeof(void *) <= sizeof(kmp_uintptr_t));
  // Make sure kmp_uintptr_t is enough to store addresses.

  descr.size_aligned = size;
  descr.size_allocated =
      descr.size_aligned + sizeof(kmp_mem_descr_t) + alignment;

#if KMP_DEBUG
  descr.ptr_allocated = _malloc_src_loc(descr.size_allocated, _file_, _line_);
#else
  descr.ptr_allocated = malloc_src_loc(descr.size_allocated KMP_SRC_LOC_PARM);
#endif
  KE_TRACE(10, ("   malloc( %d ) returned %p\n", (int)descr.size_allocated,
                descr.ptr_allocated));
  if (descr.ptr_allocated == NULL) {
    KMP_FATAL(OutOfHeapMemory);
  }

  addr_allocated = (kmp_uintptr_t)descr.ptr_allocated;
  addr_aligned =
      (addr_allocated + sizeof(kmp_mem_descr_t) + alignment) & ~(alignment - 1);
  addr_descr = addr_aligned - sizeof(kmp_mem_descr_t);

  descr.ptr_aligned = (void *)addr_aligned;

  KE_TRACE(26, ("   ___kmp_allocate_align: "
                "ptr_allocated=%p, size_allocated=%d, "
                "ptr_aligned=%p, size_aligned=%d\n",
                descr.ptr_allocated, (int)descr.size_allocated,
                descr.ptr_aligned, (int)descr.size_aligned));

  KMP_DEBUG_ASSERT(addr_allocated <= addr_descr);
  KMP_DEBUG_ASSERT(addr_descr + sizeof(kmp_mem_descr_t) == addr_aligned);
  KMP_DEBUG_ASSERT(addr_aligned + descr.size_aligned <=
                   addr_allocated + descr.size_allocated);
  KMP_DEBUG_ASSERT(addr_aligned % alignment == 0);
#ifdef KMP_DEBUG
  memset(descr.ptr_allocated, 0xEF, descr.size_allocated);
// Fill allocated memory block with 0xEF.
#endif
  memset(descr.ptr_aligned, 0x00, descr.size_aligned);
  // Fill the aligned memory block (which is intended for using by caller) with
  // 0x00. Do not
  // put this filling under KMP_DEBUG condition! Many callers expect zeroed
  // memory. (Padding
  // bytes remain filled with 0xEF in debugging library.)
  *((kmp_mem_descr_t *)addr_descr) = descr;

  KMP_MB();

  KE_TRACE(25, ("<- ___kmp_allocate_align() returns %p\n", descr.ptr_aligned));
  return descr.ptr_aligned;
} // func ___kmp_allocate_align

/* Allocate memory on cache line boundary, fill allocated memory with 0x00.
   Do not call this func directly! Use __kmp_allocate macro instead.
   NULL is NEVER returned, __kmp_abort() is called in case of memory allocation
   error. Must use __kmp_free when freeing memory allocated by this routine! */
void *___kmp_allocate(size_t size KMP_SRC_LOC_DECL) {
  void *ptr;
  KE_TRACE(25, ("-> __kmp_allocate( %d ) called from %s:%d\n",
                (int)size KMP_SRC_LOC_PARM));
  ptr = ___kmp_allocate_align(size, __kmp_align_alloc KMP_SRC_LOC_PARM);
  KE_TRACE(25, ("<- __kmp_allocate() returns %p\n", ptr));
  return ptr;
} // func ___kmp_allocate

/* Allocate memory on page boundary, fill allocated memory with 0x00.
   Does not call this func directly! Use __kmp_page_allocate macro instead.
   NULL is NEVER returned, __kmp_abort() is called in case of memory allocation
   error. Must use __kmp_free when freeing memory allocated by this routine! */
void *___kmp_page_allocate(size_t size KMP_SRC_LOC_DECL) {
  int page_size = 8 * 1024;
  void *ptr;

  KE_TRACE(25, ("-> __kmp_page_allocate( %d ) called from %s:%d\n",
                (int)size KMP_SRC_LOC_PARM));
  ptr = ___kmp_allocate_align(size, page_size KMP_SRC_LOC_PARM);
  KE_TRACE(25, ("<- __kmp_page_allocate( %d ) returns %p\n", (int)size, ptr));
  return ptr;
} // ___kmp_page_allocate

/* Free memory allocated by __kmp_allocate() and __kmp_page_allocate().
   In debug mode, fill the memory block with 0xEF before call to free(). */
void ___kmp_free(void *ptr KMP_SRC_LOC_DECL) {
  kmp_mem_descr_t descr;
#if KMP_DEBUG
  kmp_uintptr_t addr_allocated; // Address returned by malloc().
  kmp_uintptr_t addr_aligned; // Aligned address passed by caller.
#endif
  KE_TRACE(25,
           ("-> __kmp_free( %p ) called from %s:%d\n", ptr KMP_SRC_LOC_PARM));
  KMP_ASSERT(ptr != NULL);

  descr = *(kmp_mem_descr_t *)((kmp_uintptr_t)ptr - sizeof(kmp_mem_descr_t));

  KE_TRACE(26, ("   __kmp_free:     "
                "ptr_allocated=%p, size_allocated=%d, "
                "ptr_aligned=%p, size_aligned=%d\n",
                descr.ptr_allocated, (int)descr.size_allocated,
                descr.ptr_aligned, (int)descr.size_aligned));
#if KMP_DEBUG
  addr_allocated = (kmp_uintptr_t)descr.ptr_allocated;
  addr_aligned = (kmp_uintptr_t)descr.ptr_aligned;
  KMP_DEBUG_ASSERT(addr_aligned % CACHE_LINE == 0);
  KMP_DEBUG_ASSERT(descr.ptr_aligned == ptr);
  KMP_DEBUG_ASSERT(addr_allocated + sizeof(kmp_mem_descr_t) <= addr_aligned);
  KMP_DEBUG_ASSERT(descr.size_aligned < descr.size_allocated);
  KMP_DEBUG_ASSERT(addr_aligned + descr.size_aligned <=
                   addr_allocated + descr.size_allocated);
  memset(descr.ptr_allocated, 0xEF, descr.size_allocated);
// Fill memory block with 0xEF, it helps catch using freed memory.
#endif

#ifndef LEAK_MEMORY
  KE_TRACE(10, ("   free( %p )\n", descr.ptr_allocated));
#ifdef KMP_DEBUG
  _free_src_loc(descr.ptr_allocated, _file_, _line_);
#else
  free_src_loc(descr.ptr_allocated KMP_SRC_LOC_PARM);
#endif
#endif
  KMP_MB();
  KE_TRACE(25, ("<- __kmp_free() returns\n"));
} // func ___kmp_free

#if USE_FAST_MEMORY == 3
// Allocate fast memory by first scanning the thread's free lists
// If a chunk the right size exists, grab it off the free list.
// Otherwise allocate normally using kmp_thread_malloc.

// AC: How to choose the limit? Just get 16 for now...
#define KMP_FREE_LIST_LIMIT 16

// Always use 128 bytes for determining buckets for caching memory blocks
#define DCACHE_LINE 128

void *___kmp_fast_allocate(kmp_info_t *this_thr, size_t size KMP_SRC_LOC_DECL) {
  void *ptr;
  size_t num_lines, idx;
  int index;
  void *alloc_ptr;
  size_t alloc_size;
  kmp_mem_descr_t *descr;

  KE_TRACE(25, ("-> __kmp_fast_allocate( T#%d, %d ) called from %s:%d\n",
                __kmp_gtid_from_thread(this_thr), (int)size KMP_SRC_LOC_PARM));

  num_lines = (size + DCACHE_LINE - 1) / DCACHE_LINE;
  idx = num_lines - 1;
  KMP_DEBUG_ASSERT(idx >= 0);
  if (idx < 2) {
    index = 0; // idx is [ 0, 1 ], use first free list
    num_lines = 2; // 1, 2 cache lines or less than cache line
  } else if ((idx >>= 2) == 0) {
    index = 1; // idx is [ 2, 3 ], use second free list
    num_lines = 4; // 3, 4 cache lines
  } else if ((idx >>= 2) == 0) {
    index = 2; // idx is [ 4, 15 ], use third free list
    num_lines = 16; // 5, 6, ..., 16 cache lines
  } else if ((idx >>= 2) == 0) {
    index = 3; // idx is [ 16, 63 ], use fourth free list
    num_lines = 64; // 17, 18, ..., 64 cache lines
  } else {
    goto alloc_call; // 65 or more cache lines ( > 8KB ), don't use free lists
  }

  ptr = this_thr->th.th_free_lists[index].th_free_list_self;
  if (ptr != NULL) {
    // pop the head of no-sync free list
    this_thr->th.th_free_lists[index].th_free_list_self = *((void **)ptr);
    KMP_DEBUG_ASSERT(this_thr == ((kmp_mem_descr_t *)((kmp_uintptr_t)ptr -
                                                      sizeof(kmp_mem_descr_t)))
                                     ->ptr_aligned);
    goto end;
  }
  ptr = TCR_SYNC_PTR(this_thr->th.th_free_lists[index].th_free_list_sync);
  if (ptr != NULL) {
    // no-sync free list is empty, use sync free list (filled in by other
    // threads only)
    // pop the head of the sync free list, push NULL instead
    while (!KMP_COMPARE_AND_STORE_PTR(
        &this_thr->th.th_free_lists[index].th_free_list_sync, ptr, nullptr)) {
      KMP_CPU_PAUSE();
      ptr = TCR_SYNC_PTR(this_thr->th.th_free_lists[index].th_free_list_sync);
    }
    // push the rest of chain into no-sync free list (can be NULL if there was
    // the only block)
    this_thr->th.th_free_lists[index].th_free_list_self = *((void **)ptr);
    KMP_DEBUG_ASSERT(this_thr == ((kmp_mem_descr_t *)((kmp_uintptr_t)ptr -
                                                      sizeof(kmp_mem_descr_t)))
                                     ->ptr_aligned);
    goto end;
  }

alloc_call:
  // haven't found block in the free lists, thus allocate it
  size = num_lines * DCACHE_LINE;

  alloc_size = size + sizeof(kmp_mem_descr_t) + DCACHE_LINE;
  KE_TRACE(25, ("__kmp_fast_allocate: T#%d Calling __kmp_thread_malloc with "
                "alloc_size %d\n",
                __kmp_gtid_from_thread(this_thr), alloc_size));
  alloc_ptr = bget(this_thr, (bufsize)alloc_size);

  // align ptr to DCACHE_LINE
  ptr = (void *)((((kmp_uintptr_t)alloc_ptr) + sizeof(kmp_mem_descr_t) +
                  DCACHE_LINE) &
                 ~(DCACHE_LINE - 1));
  descr = (kmp_mem_descr_t *)(((kmp_uintptr_t)ptr) - sizeof(kmp_mem_descr_t));

  descr->ptr_allocated = alloc_ptr; // remember allocated pointer
  // we don't need size_allocated
  descr->ptr_aligned = (void *)this_thr; // remember allocating thread
  // (it is already saved in bget buffer,
  // but we may want to use another allocator in future)
  descr->size_aligned = size;

end:
  KE_TRACE(25, ("<- __kmp_fast_allocate( T#%d ) returns %p\n",
                __kmp_gtid_from_thread(this_thr), ptr));
  return ptr;
} // func __kmp_fast_allocate

// Free fast memory and place it on the thread's free list if it is of
// the correct size.
void ___kmp_fast_free(kmp_info_t *this_thr, void *ptr KMP_SRC_LOC_DECL) {
  kmp_mem_descr_t *descr;
  kmp_info_t *alloc_thr;
  size_t size;
  size_t idx;
  int index;

  KE_TRACE(25, ("-> __kmp_fast_free( T#%d, %p ) called from %s:%d\n",
                __kmp_gtid_from_thread(this_thr), ptr KMP_SRC_LOC_PARM));
  KMP_ASSERT(ptr != NULL);

  descr = (kmp_mem_descr_t *)(((kmp_uintptr_t)ptr) - sizeof(kmp_mem_descr_t));

  KE_TRACE(26, ("   __kmp_fast_free:     size_aligned=%d\n",
                (int)descr->size_aligned));

  size = descr->size_aligned; // 2, 4, 16, 64, 65, 66, ... cache lines

  idx = DCACHE_LINE * 2; // 2 cache lines is minimal size of block
  if (idx == size) {
    index = 0; // 2 cache lines
  } else if ((idx <<= 1) == size) {
    index = 1; // 4 cache lines
  } else if ((idx <<= 2) == size) {
    index = 2; // 16 cache lines
  } else if ((idx <<= 2) == size) {
    index = 3; // 64 cache lines
  } else {
    KMP_DEBUG_ASSERT(size > DCACHE_LINE * 64);
    goto free_call; // 65 or more cache lines ( > 8KB )
  }

  alloc_thr = (kmp_info_t *)descr->ptr_aligned; // get thread owning the block
  if (alloc_thr == this_thr) {
    // push block to self no-sync free list, linking previous head (LIFO)
    *((void **)ptr) = this_thr->th.th_free_lists[index].th_free_list_self;
    this_thr->th.th_free_lists[index].th_free_list_self = ptr;
  } else {
    void *head = this_thr->th.th_free_lists[index].th_free_list_other;
    if (head == NULL) {
      // Create new free list
      this_thr->th.th_free_lists[index].th_free_list_other = ptr;
      *((void **)ptr) = NULL; // mark the tail of the list
      descr->size_allocated = (size_t)1; // head of the list keeps its length
    } else {
      // need to check existed "other" list's owner thread and size of queue
      kmp_mem_descr_t *dsc =
          (kmp_mem_descr_t *)((char *)head - sizeof(kmp_mem_descr_t));
      // allocating thread, same for all queue nodes
      kmp_info_t *q_th = (kmp_info_t *)(dsc->ptr_aligned);
      size_t q_sz =
          dsc->size_allocated + 1; // new size in case we add current task
      if (q_th == alloc_thr && q_sz <= KMP_FREE_LIST_LIMIT) {
        // we can add current task to "other" list, no sync needed
        *((void **)ptr) = head;
        descr->size_allocated = q_sz;
        this_thr->th.th_free_lists[index].th_free_list_other = ptr;
      } else {
        // either queue blocks owner is changing or size limit exceeded
        // return old queue to allocating thread (q_th) synchronously,
        // and start new list for alloc_thr's tasks
        void *old_ptr;
        void *tail = head;
        void *next = *((void **)head);
        while (next != NULL) {
          KMP_DEBUG_ASSERT(
              // queue size should decrease by 1 each step through the list
              ((kmp_mem_descr_t *)((char *)next - sizeof(kmp_mem_descr_t)))
                      ->size_allocated +
                  1 ==
              ((kmp_mem_descr_t *)((char *)tail - sizeof(kmp_mem_descr_t)))
                  ->size_allocated);
          tail = next; // remember tail node
          next = *((void **)next);
        }
        KMP_DEBUG_ASSERT(q_th != NULL);
        // push block to owner's sync free list
        old_ptr = TCR_PTR(q_th->th.th_free_lists[index].th_free_list_sync);
        /* the next pointer must be set before setting free_list to ptr to avoid
           exposing a broken list to other threads, even for an instant. */
        *((void **)tail) = old_ptr;

        while (!KMP_COMPARE_AND_STORE_PTR(
            &q_th->th.th_free_lists[index].th_free_list_sync, old_ptr, head)) {
          KMP_CPU_PAUSE();
          old_ptr = TCR_PTR(q_th->th.th_free_lists[index].th_free_list_sync);
          *((void **)tail) = old_ptr;
        }

        // start new list of not-selt tasks
        this_thr->th.th_free_lists[index].th_free_list_other = ptr;
        *((void **)ptr) = NULL;
        descr->size_allocated = (size_t)1; // head of queue keeps its length
      }
    }
  }
  goto end;

free_call:
  KE_TRACE(25, ("__kmp_fast_free: T#%d Calling __kmp_thread_free for size %d\n",
                __kmp_gtid_from_thread(this_thr), size));
  __kmp_bget_dequeue(this_thr); /* Release any queued buffers */
  brel(this_thr, descr->ptr_allocated);

end:
  KE_TRACE(25, ("<- __kmp_fast_free() returns\n"));

} // func __kmp_fast_free

// Initialize the thread free lists related to fast memory
// Only do this when a thread is initially created.
void __kmp_initialize_fast_memory(kmp_info_t *this_thr) {
  KE_TRACE(10, ("__kmp_initialize_fast_memory: Called from th %p\n", this_thr));

  memset(this_thr->th.th_free_lists, 0, NUM_LISTS * sizeof(kmp_free_list_t));
}

// Free the memory in the thread free lists related to fast memory
// Only do this when a thread is being reaped (destroyed).
void __kmp_free_fast_memory(kmp_info_t *th) {
  // Suppose we use BGET underlying allocator, walk through its structures...
  int bin;
  thr_data_t *thr = get_thr_data(th);
  void **lst = NULL;

  KE_TRACE(
      5, ("__kmp_free_fast_memory: Called T#%d\n", __kmp_gtid_from_thread(th)));

  __kmp_bget_dequeue(th); // Release any queued buffers

  // Dig through free lists and extract all allocated blocks
  for (bin = 0; bin < MAX_BGET_BINS; ++bin) {
    bfhead_t *b = thr->freelist[bin].ql.flink;
    while (b != &thr->freelist[bin]) {
      if ((kmp_uintptr_t)b->bh.bb.bthr & 1) { // the buffer is allocated address
        *((void **)b) =
            lst; // link the list (override bthr, but keep flink yet)
        lst = (void **)b; // push b into lst
      }
      b = b->ql.flink; // get next buffer
    }
  }
  while (lst != NULL) {
    void *next = *lst;
    KE_TRACE(10, ("__kmp_free_fast_memory: freeing %p, next=%p th %p (%d)\n",
                  lst, next, th, __kmp_gtid_from_thread(th)));
    (*thr->relfcn)(lst);
#if BufStats
    // count blocks to prevent problems in __kmp_finalize_bget()
    thr->numprel++; /* Nr of expansion block releases */
    thr->numpblk--; /* Total number of blocks */
#endif
    lst = (void **)next;
  }

  KE_TRACE(
      5, ("__kmp_free_fast_memory: Freed T#%d\n", __kmp_gtid_from_thread(th)));
}

#endif // USE_FAST_MEMORY
