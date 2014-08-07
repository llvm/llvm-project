/*
 * kmp_lock.h -- lock header file
 * $Revision: 42810 $
 * $Date: 2013-11-07 12:06:33 -0600 (Thu, 07 Nov 2013) $
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef KMP_LOCK_H
#define KMP_LOCK_H

#include <limits.h>    // CHAR_BIT
#include <stddef.h>    // offsetof

#include "kmp_os.h"
#include "kmp_debug.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ----------------------------------------------------------------------------
// Have to copy these definitions from kmp.h because kmp.h cannot be included
// due to circular dependencies.  Will undef these at end of file.

#define KMP_PAD(type, sz)     (sizeof(type) + (sz - ((sizeof(type) - 1) % (sz)) - 1))
#define KMP_GTID_DNE (-2)

// Forward declaration of ident and ident_t

struct ident;
typedef struct ident ident_t;

// End of copied code.
// ----------------------------------------------------------------------------

//
// We need to know the size of the area we can assume that the compiler(s)
// allocated for obects of type omp_lock_t and omp_nest_lock_t.  The Intel
// compiler always allocates a pointer-sized area, as does visual studio.
//
// gcc however, only allocates 4 bytes for regular locks, even on 64-bit
// intel archs.  It allocates at least 8 bytes for nested lock (more on
// recent versions), but we are bounded by the pointer-sized chunks that
// the Intel compiler allocates.
//

#if KMP_OS_LINUX && defined(KMP_GOMP_COMPAT)
# define OMP_LOCK_T_SIZE        sizeof(int)
# define OMP_NEST_LOCK_T_SIZE   sizeof(void *)
#else
# define OMP_LOCK_T_SIZE        sizeof(void *)
# define OMP_NEST_LOCK_T_SIZE   sizeof(void *)
#endif

//
// The Intel compiler allocates a 32-byte chunk for a critical section.
// Both gcc and visual studio only allocate enough space for a pointer.
// Sometimes we know that the space was allocated by the Intel compiler.
//
#define OMP_CRITICAL_SIZE       sizeof(void *)
#define INTEL_CRITICAL_SIZE     32

//
// lock flags
//
typedef kmp_uint32 kmp_lock_flags_t;

#define kmp_lf_critical_section 1

//
// When a lock table is used, the indices are of kmp_lock_index_t
//
typedef kmp_uint32 kmp_lock_index_t;

//
// When memory allocated for locks are on the lock pool (free list),
// it is treated as structs of this type.
//
struct kmp_lock_pool {
    union kmp_user_lock *next;
    kmp_lock_index_t index;
};

typedef struct kmp_lock_pool kmp_lock_pool_t;


extern void __kmp_validate_locks( void );


// ----------------------------------------------------------------------------
//
//  There are 5 lock implementations:
//
//       1. Test and set locks.
//       2. futex locks (Linux* OS on x86 and Intel(R) Many Integrated Core architecture)
//       3. Ticket (Lamport bakery) locks.
//       4. Queuing locks (with separate spin fields).
//       5. DRPA (Dynamically Reconfigurable Distributed Polling Area) locks
//
//   and 3 lock purposes:
//
//       1. Bootstrap locks -- Used for a few locks available at library startup-shutdown time.
//          These do not require non-negative global thread ID's.
//       2. Internal RTL locks -- Used everywhere else in the RTL
//       3. User locks (includes critical sections)
//
// ----------------------------------------------------------------------------


// ============================================================================
// Lock implementations.
// ============================================================================


// ----------------------------------------------------------------------------
// Test and set locks.
//
// Non-nested test and set locks differ from the other lock kinds (except
// futex) in that we use the memory allocated by the compiler for the lock,
// rather than a pointer to it.
//
// On lin32, lin_32e, and win_32, the space allocated may be as small as 4
// bytes, so we have to use a lock table for nested locks, and avoid accessing
// the depth_locked field for non-nested locks.
//
// Information normally available to the tools, such as lock location,
// lock usage (normal lock vs. critical section), etc. is not available with
// test and set locks.
// ----------------------------------------------------------------------------

struct kmp_base_tas_lock {
    volatile kmp_int32 poll;         // 0 => unlocked
                                     // locked: (gtid+1) of owning thread
    kmp_int32          depth_locked; // depth locked, for nested locks only
};

typedef struct kmp_base_tas_lock kmp_base_tas_lock_t;

union kmp_tas_lock {
    kmp_base_tas_lock_t lk;
    kmp_lock_pool_t pool;   // make certain struct is large enough
    double lk_align;        // use worst case alignment
                            // no cache line padding
};

typedef union kmp_tas_lock kmp_tas_lock_t;

//
// Static initializer for test and set lock variables. Usage:
//    kmp_tas_lock_t xlock = KMP_TAS_LOCK_INITIALIZER( xlock );
//
#define KMP_TAS_LOCK_INITIALIZER( lock ) { { 0, 0 } }

extern void __kmp_acquire_tas_lock( kmp_tas_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_tas_lock( kmp_tas_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_tas_lock( kmp_tas_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_tas_lock( kmp_tas_lock_t *lck );
extern void __kmp_destroy_tas_lock( kmp_tas_lock_t *lck );

extern void __kmp_acquire_nested_tas_lock( kmp_tas_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_nested_tas_lock( kmp_tas_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_nested_tas_lock( kmp_tas_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_nested_tas_lock( kmp_tas_lock_t *lck );
extern void __kmp_destroy_nested_tas_lock( kmp_tas_lock_t *lck );


#if KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM)

// ----------------------------------------------------------------------------
// futex locks.  futex locks are only available on Linux* OS.
//
// Like non-nested test and set lock, non-nested futex locks use the memory
// allocated by the compiler for the lock, rather than a pointer to it.
//
// Information normally available to the tools, such as lock location,
// lock usage (normal lock vs. critical section), etc. is not available with
// test and set locks.  With non-nested futex locks, the lock owner is not
// even available.
// ----------------------------------------------------------------------------

struct kmp_base_futex_lock {
    volatile kmp_int32 poll;         // 0 => unlocked
                                     // 2*(gtid+1) of owning thread, 0 if unlocked
                                     // locked: (gtid+1) of owning thread
    kmp_int32          depth_locked; // depth locked, for nested locks only
};

typedef struct kmp_base_futex_lock kmp_base_futex_lock_t;

union kmp_futex_lock {
    kmp_base_futex_lock_t lk;
    kmp_lock_pool_t pool;   // make certain struct is large enough
    double lk_align;        // use worst case alignment
                            // no cache line padding
};

typedef union kmp_futex_lock kmp_futex_lock_t;

//
// Static initializer for futex lock variables. Usage:
//    kmp_futex_lock_t xlock = KMP_FUTEX_LOCK_INITIALIZER( xlock );
//
#define KMP_FUTEX_LOCK_INITIALIZER( lock ) { { 0, 0 } }

extern void __kmp_acquire_futex_lock( kmp_futex_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_futex_lock( kmp_futex_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_futex_lock( kmp_futex_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_futex_lock( kmp_futex_lock_t *lck );
extern void __kmp_destroy_futex_lock( kmp_futex_lock_t *lck );

extern void __kmp_acquire_nested_futex_lock( kmp_futex_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_nested_futex_lock( kmp_futex_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_nested_futex_lock( kmp_futex_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_nested_futex_lock( kmp_futex_lock_t *lck );
extern void __kmp_destroy_nested_futex_lock( kmp_futex_lock_t *lck );

#endif // KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM)


// ----------------------------------------------------------------------------
// Ticket locks.
// ----------------------------------------------------------------------------

struct kmp_base_ticket_lock {
    // `initialized' must be the first entry in the lock data structure!
    volatile union kmp_ticket_lock * initialized;  // points to the lock union if in initialized state
    ident_t const *     location;     // Source code location of omp_init_lock().
    volatile kmp_uint32 next_ticket;  // ticket number to give to next thread which acquires
    volatile kmp_uint32 now_serving;  // ticket number for thread which holds the lock
    volatile kmp_int32  owner_id;     // (gtid+1) of owning thread, 0 if unlocked
    kmp_int32           depth_locked; // depth locked, for nested locks only
    kmp_lock_flags_t    flags;        // lock specifics, e.g. critical section lock
};

typedef struct kmp_base_ticket_lock kmp_base_ticket_lock_t;

union KMP_ALIGN_CACHE kmp_ticket_lock {
    kmp_base_ticket_lock_t lk;       // This field must be first to allow static initializing.
    kmp_lock_pool_t pool;
    double                 lk_align; // use worst case alignment
    char                   lk_pad[ KMP_PAD( kmp_base_ticket_lock_t, CACHE_LINE ) ];
};

typedef union kmp_ticket_lock kmp_ticket_lock_t;

//
// Static initializer for simple ticket lock variables. Usage:
//    kmp_ticket_lock_t xlock = KMP_TICKET_LOCK_INITIALIZER( xlock );
// Note the macro argument. It is important to make var properly initialized.
//
#define KMP_TICKET_LOCK_INITIALIZER( lock ) { { (kmp_ticket_lock_t *) & (lock), NULL, 0, 0, 0, -1 } }

extern void __kmp_acquire_ticket_lock( kmp_ticket_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_ticket_lock( kmp_ticket_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_ticket_lock_with_cheks( kmp_ticket_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_ticket_lock( kmp_ticket_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_ticket_lock( kmp_ticket_lock_t *lck );
extern void __kmp_destroy_ticket_lock( kmp_ticket_lock_t *lck );

extern void __kmp_acquire_nested_ticket_lock( kmp_ticket_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_nested_ticket_lock( kmp_ticket_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_nested_ticket_lock( kmp_ticket_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_nested_ticket_lock( kmp_ticket_lock_t *lck );
extern void __kmp_destroy_nested_ticket_lock( kmp_ticket_lock_t *lck );


// ----------------------------------------------------------------------------
// Queuing locks.
// ----------------------------------------------------------------------------

#if KMP_USE_ADAPTIVE_LOCKS

struct kmp_adaptive_lock;

typedef struct kmp_adaptive_lock kmp_adaptive_lock_t;

#if KMP_DEBUG_ADAPTIVE_LOCKS

struct kmp_adaptive_lock_statistics {
    /* So we can get stats from locks that haven't been destroyed. */
    kmp_adaptive_lock_t * next;
    kmp_adaptive_lock_t * prev;

    /* Other statistics */
    kmp_uint32 successfulSpeculations;
    kmp_uint32 hardFailedSpeculations;
    kmp_uint32 softFailedSpeculations;
    kmp_uint32 nonSpeculativeAcquires;
    kmp_uint32 nonSpeculativeAcquireAttempts;
    kmp_uint32 lemmingYields;
};

typedef struct kmp_adaptive_lock_statistics kmp_adaptive_lock_statistics_t;

extern void __kmp_print_speculative_stats();
extern void __kmp_init_speculative_stats();

#endif // KMP_DEBUG_ADAPTIVE_LOCKS

struct kmp_adaptive_lock
{
    /* Values used for adaptivity.
     * Although these are accessed from multiple threads we don't access them atomically,
     * because if we miss updates it probably doesn't matter much. (It just affects our
     * decision about whether to try speculation on the lock).
     */
    kmp_uint32 volatile badness;
    kmp_uint32 volatile acquire_attempts;
    /* Parameters of the lock. */
    kmp_uint32 max_badness;
    kmp_uint32 max_soft_retries;

#if KMP_DEBUG_ADAPTIVE_LOCKS
    kmp_adaptive_lock_statistics_t volatile stats;
#endif
};

#endif // KMP_USE_ADAPTIVE_LOCKS


struct kmp_base_queuing_lock {

    //  `initialized' must be the first entry in the lock data structure!
    volatile union kmp_queuing_lock *initialized; // Points to the lock union if in initialized state.

    ident_t const *     location;     // Source code location of omp_init_lock().

    KMP_ALIGN( 8 )                    // tail_id  must be 8-byte aligned!

    volatile kmp_int32  tail_id;      // (gtid+1) of thread at tail of wait queue, 0 if empty
                                      // Must be no padding here since head/tail used in 8-byte CAS
    volatile kmp_int32  head_id;      // (gtid+1) of thread at head of wait queue, 0 if empty
                                      // Decl order assumes little endian
    // bakery-style lock
    volatile kmp_uint32 next_ticket;  // ticket number to give to next thread which acquires
    volatile kmp_uint32 now_serving;  // ticket number for thread which holds the lock
    volatile kmp_int32  owner_id;     // (gtid+1) of owning thread, 0 if unlocked
    kmp_int32           depth_locked; // depth locked, for nested locks only

    kmp_lock_flags_t    flags;        // lock specifics, e.g. critical section lock
#if KMP_USE_ADAPTIVE_LOCKS
    KMP_ALIGN(CACHE_LINE)
    kmp_adaptive_lock_t adaptive;     // Information for the speculative adaptive lock
#endif
};

typedef struct kmp_base_queuing_lock kmp_base_queuing_lock_t;

KMP_BUILD_ASSERT( offsetof( kmp_base_queuing_lock_t, tail_id ) % 8 == 0 );

union KMP_ALIGN_CACHE kmp_queuing_lock {
    kmp_base_queuing_lock_t lk;       // This field must be first to allow static initializing.
    kmp_lock_pool_t pool;
    double                   lk_align; // use worst case alignment
    char                     lk_pad[ KMP_PAD( kmp_base_queuing_lock_t, CACHE_LINE ) ];
};

typedef union kmp_queuing_lock kmp_queuing_lock_t;

extern void __kmp_acquire_queuing_lock( kmp_queuing_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_queuing_lock( kmp_queuing_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_queuing_lock( kmp_queuing_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_queuing_lock( kmp_queuing_lock_t *lck );
extern void __kmp_destroy_queuing_lock( kmp_queuing_lock_t *lck );

extern void __kmp_acquire_nested_queuing_lock( kmp_queuing_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_nested_queuing_lock( kmp_queuing_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_nested_queuing_lock( kmp_queuing_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_nested_queuing_lock( kmp_queuing_lock_t *lck );
extern void __kmp_destroy_nested_queuing_lock( kmp_queuing_lock_t *lck );


// ----------------------------------------------------------------------------
// DRDPA ticket locks.
// ----------------------------------------------------------------------------

struct kmp_base_drdpa_lock {
    //
    // All of the fields on the first cache line are only written when
    // initializing or reconfiguring the lock.  These are relatively rare
    // operations, so data from the first cache line will usually stay
    // resident in the cache of each thread trying to acquire the lock.
    //
    // initialized must be the first entry in the lock data structure!
    //
    KMP_ALIGN_CACHE

    volatile union kmp_drdpa_lock * initialized;    // points to the lock union if in initialized state
    ident_t const *                 location;       // Source code location of omp_init_lock().
    volatile struct kmp_lock_poll {
        kmp_uint64 poll;
    } * volatile polls;
    volatile kmp_uint64             mask;           // is 2**num_polls-1 for mod op
    kmp_uint64                      cleanup_ticket; // thread with cleanup ticket
    volatile struct kmp_lock_poll * old_polls;      // will deallocate old_polls
    kmp_uint32                      num_polls;      // must be power of 2

    //
    // next_ticket it needs to exist in a separate cache line, as it is
    // invalidated every time a thread takes a new ticket.
    //
    KMP_ALIGN_CACHE

    volatile kmp_uint64             next_ticket;

    //
    // now_serving is used to store our ticket value while we hold the lock.
    // It has a slightly different meaning in the DRDPA ticket locks (where
    // it is written by the acquiring thread) than it does in the simple
    // ticket locks (where it is written by the releasing thread).
    //
    // Since now_serving is only read an written in the critical section,
    // it is non-volatile, but it needs to exist on a separate cache line,
    // as it is invalidated at every lock acquire.
    //
    // Likewise, the vars used for nested locks (owner_id and depth_locked)
    // are only written by the thread owning the lock, so they are put in
    // this cache line.  owner_id is read by other threads, so it must be
    // declared volatile.
    //
    KMP_ALIGN_CACHE

    kmp_uint64                      now_serving;    // doesn't have to be volatile
    volatile kmp_uint32             owner_id;       // (gtid+1) of owning thread, 0 if unlocked
    kmp_int32                       depth_locked;   // depth locked
    kmp_lock_flags_t                flags;          // lock specifics, e.g. critical section lock
};

typedef struct kmp_base_drdpa_lock kmp_base_drdpa_lock_t;

union KMP_ALIGN_CACHE kmp_drdpa_lock {
    kmp_base_drdpa_lock_t lk;       // This field must be first to allow static initializing. */
    kmp_lock_pool_t pool;
    double                lk_align; // use worst case alignment
    char                  lk_pad[ KMP_PAD( kmp_base_drdpa_lock_t, CACHE_LINE ) ];
};

typedef union kmp_drdpa_lock kmp_drdpa_lock_t;

extern void __kmp_acquire_drdpa_lock( kmp_drdpa_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_drdpa_lock( kmp_drdpa_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_drdpa_lock( kmp_drdpa_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_drdpa_lock( kmp_drdpa_lock_t *lck );
extern void __kmp_destroy_drdpa_lock( kmp_drdpa_lock_t *lck );

extern void __kmp_acquire_nested_drdpa_lock( kmp_drdpa_lock_t *lck, kmp_int32 gtid );
extern int __kmp_test_nested_drdpa_lock( kmp_drdpa_lock_t *lck, kmp_int32 gtid );
extern void __kmp_release_nested_drdpa_lock( kmp_drdpa_lock_t *lck, kmp_int32 gtid );
extern void __kmp_init_nested_drdpa_lock( kmp_drdpa_lock_t *lck );
extern void __kmp_destroy_nested_drdpa_lock( kmp_drdpa_lock_t *lck );


// ============================================================================
// Lock purposes.
// ============================================================================


// ----------------------------------------------------------------------------
// Bootstrap locks.
// ----------------------------------------------------------------------------

// Bootstrap locks -- very few locks used at library initialization time.
// Bootstrap locks are currently implemented as ticket locks.
// They could also be implemented as test and set lock, but cannot be
// implemented with other lock kinds as they require gtids which are not
// available at initialization time.

typedef kmp_ticket_lock_t kmp_bootstrap_lock_t;

#define KMP_BOOTSTRAP_LOCK_INITIALIZER( lock ) KMP_TICKET_LOCK_INITIALIZER( (lock) )

static inline void
__kmp_acquire_bootstrap_lock( kmp_bootstrap_lock_t *lck )
{
    __kmp_acquire_ticket_lock( lck, KMP_GTID_DNE );
}

static inline int
__kmp_test_bootstrap_lock( kmp_bootstrap_lock_t *lck )
{
    return __kmp_test_ticket_lock( lck, KMP_GTID_DNE );
}

static inline void
__kmp_release_bootstrap_lock( kmp_bootstrap_lock_t *lck )
{
    __kmp_release_ticket_lock( lck, KMP_GTID_DNE );
}

static inline void
__kmp_init_bootstrap_lock( kmp_bootstrap_lock_t *lck )
{
    __kmp_init_ticket_lock( lck );
}

static inline void
__kmp_destroy_bootstrap_lock( kmp_bootstrap_lock_t *lck )
{
    __kmp_destroy_ticket_lock( lck );
}


// ----------------------------------------------------------------------------
// Internal RTL locks.
// ----------------------------------------------------------------------------

//
// Internal RTL locks are also implemented as ticket locks, for now.
//
// FIXME - We should go through and figure out which lock kind works best for
// each internal lock, and use the type declaration and function calls for
// that explicit lock kind (and get rid of this section).
//

typedef kmp_ticket_lock_t kmp_lock_t;

static inline void
__kmp_acquire_lock( kmp_lock_t *lck, kmp_int32 gtid )
{
    __kmp_acquire_ticket_lock( lck, gtid );
}

static inline int
__kmp_test_lock( kmp_lock_t *lck, kmp_int32 gtid )
{
    return __kmp_test_ticket_lock( lck, gtid );
}

static inline void
__kmp_release_lock( kmp_lock_t *lck, kmp_int32 gtid )
{
    __kmp_release_ticket_lock( lck, gtid );
}

static inline void
__kmp_init_lock( kmp_lock_t *lck )
{
    __kmp_init_ticket_lock( lck );
}

static inline void
__kmp_destroy_lock( kmp_lock_t *lck )
{
    __kmp_destroy_ticket_lock( lck );
}


// ----------------------------------------------------------------------------
// User locks.
// ----------------------------------------------------------------------------

//
// Do not allocate objects of type union kmp_user_lock!!!
// This will waste space unless __kmp_user_lock_kind == lk_drdpa.
// Instead, check the value of __kmp_user_lock_kind and allocate objects of
// the type of the appropriate union member, and cast their addresses to
// kmp_user_lock_p.
//

enum kmp_lock_kind {
    lk_default = 0,
    lk_tas,
#if KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM)
    lk_futex,
#endif
    lk_ticket,
    lk_queuing,
    lk_drdpa,
#if KMP_USE_ADAPTIVE_LOCKS
    lk_adaptive
#endif // KMP_USE_ADAPTIVE_LOCKS
};

typedef enum kmp_lock_kind kmp_lock_kind_t;

extern kmp_lock_kind_t __kmp_user_lock_kind;

union kmp_user_lock {
    kmp_tas_lock_t     tas;
#if KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM)
    kmp_futex_lock_t   futex;
#endif
    kmp_ticket_lock_t  ticket;
    kmp_queuing_lock_t queuing;
    kmp_drdpa_lock_t   drdpa;
#if KMP_USE_ADAPTIVE_LOCKS
    kmp_adaptive_lock_t     adaptive;
#endif // KMP_USE_ADAPTIVE_LOCKS
    kmp_lock_pool_t    pool;
};

typedef union kmp_user_lock *kmp_user_lock_p;

extern size_t __kmp_base_user_lock_size;
extern size_t __kmp_user_lock_size;

extern kmp_int32 ( *__kmp_get_user_lock_owner_ )( kmp_user_lock_p lck );

static inline kmp_int32
__kmp_get_user_lock_owner( kmp_user_lock_p lck )
{
    KMP_DEBUG_ASSERT( __kmp_get_user_lock_owner_ != NULL );
    return ( *__kmp_get_user_lock_owner_ )( lck );
}

extern void ( *__kmp_acquire_user_lock_with_checks_ )( kmp_user_lock_p lck, kmp_int32 gtid );

#if KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM)

#define __kmp_acquire_user_lock_with_checks(lck,gtid)                                           \
    if (__kmp_user_lock_kind == lk_tas) {                                                       \
        if ( __kmp_env_consistency_check ) {                                                    \
            char const * const func = "omp_set_lock";                                           \
            if ( ( sizeof ( kmp_tas_lock_t ) <= OMP_LOCK_T_SIZE )                               \
                && lck->tas.lk.depth_locked != -1 ) {                                           \
                KMP_FATAL( LockNestableUsedAsSimple, func );                                    \
            }                                                                                   \
            if ( ( gtid >= 0 ) && ( lck->tas.lk.poll - 1 == gtid ) ) {                          \
                KMP_FATAL( LockIsAlreadyOwned, func );                                          \
            }                                                                                   \
        }                                                                                       \
        if ( ( lck->tas.lk.poll != 0 ) ||                                                       \
          ( ! KMP_COMPARE_AND_STORE_ACQ32( &(lck->tas.lk.poll), 0, gtid + 1 ) )  ) {            \
            kmp_uint32 spins;                                                                   \
            KMP_FSYNC_PREPARE( lck );                                                           \
            KMP_INIT_YIELD( spins );                                                            \
            if ( TCR_4(__kmp_nth) > (__kmp_avail_proc ? __kmp_avail_proc : __kmp_xproc) ) {     \
                KMP_YIELD( TRUE );                                                              \
            } else {                                                                            \
                KMP_YIELD_SPIN( spins );                                                        \
            }                                                                                   \
            while ( ( lck->tas.lk.poll != 0 ) ||                                                \
              ( ! KMP_COMPARE_AND_STORE_ACQ32( &(lck->tas.lk.poll), 0, gtid + 1 ) )  )  {       \
                if ( TCR_4(__kmp_nth) > (__kmp_avail_proc ? __kmp_avail_proc : __kmp_xproc) ) { \
                    KMP_YIELD( TRUE );                                                          \
                } else {                                                                        \
                    KMP_YIELD_SPIN( spins );                                                    \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        KMP_FSYNC_ACQUIRED( lck );                                                              \
    } else {                                                                                    \
        KMP_DEBUG_ASSERT( __kmp_acquire_user_lock_with_checks_ != NULL );                       \
        ( *__kmp_acquire_user_lock_with_checks_ )( lck, gtid );                                 \
    }

#else
static inline void
__kmp_acquire_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    KMP_DEBUG_ASSERT( __kmp_acquire_user_lock_with_checks_ != NULL );
    ( *__kmp_acquire_user_lock_with_checks_ )( lck, gtid );
}
#endif

extern int ( *__kmp_test_user_lock_with_checks_ )( kmp_user_lock_p lck, kmp_int32 gtid );

#if KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_ARCH_ARM)

#include "kmp_i18n.h"                       /* AC: KMP_FATAL definition */
extern int __kmp_env_consistency_check;     /* AC: copy from kmp.h here */
static inline int
__kmp_test_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    if ( __kmp_user_lock_kind == lk_tas ) {
        if ( __kmp_env_consistency_check ) {
            char const * const func = "omp_test_lock";
            if ( ( sizeof ( kmp_tas_lock_t ) <= OMP_LOCK_T_SIZE )
                && lck->tas.lk.depth_locked != -1 ) {
                KMP_FATAL( LockNestableUsedAsSimple, func );
            }
        }
        return ( ( lck->tas.lk.poll == 0 ) &&
          KMP_COMPARE_AND_STORE_ACQ32( &(lck->tas.lk.poll), 0, gtid + 1 ) );
    } else {
        KMP_DEBUG_ASSERT( __kmp_test_user_lock_with_checks_ != NULL );
        return ( *__kmp_test_user_lock_with_checks_ )( lck, gtid );
    }
}
#else
static inline int
__kmp_test_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    KMP_DEBUG_ASSERT( __kmp_test_user_lock_with_checks_ != NULL );
    return ( *__kmp_test_user_lock_with_checks_ )( lck, gtid );
}
#endif

extern void ( *__kmp_release_user_lock_with_checks_ )( kmp_user_lock_p lck, kmp_int32 gtid );

static inline void
__kmp_release_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    KMP_DEBUG_ASSERT( __kmp_release_user_lock_with_checks_ != NULL );
    ( *__kmp_release_user_lock_with_checks_ ) ( lck, gtid );
}

extern void ( *__kmp_init_user_lock_with_checks_ )( kmp_user_lock_p lck );

static inline void
__kmp_init_user_lock_with_checks( kmp_user_lock_p lck )
{
    KMP_DEBUG_ASSERT( __kmp_init_user_lock_with_checks_ != NULL );
    ( *__kmp_init_user_lock_with_checks_ )( lck );
}

//
// We need a non-checking version of destroy lock for when the RTL is
// doing the cleanup as it can't always tell if the lock is nested or not.
//
extern void ( *__kmp_destroy_user_lock_ )( kmp_user_lock_p lck );

static inline void
__kmp_destroy_user_lock( kmp_user_lock_p lck )
{
    KMP_DEBUG_ASSERT( __kmp_destroy_user_lock_ != NULL );
    ( *__kmp_destroy_user_lock_ )( lck );
}

extern void ( *__kmp_destroy_user_lock_with_checks_ )( kmp_user_lock_p lck );

static inline void
__kmp_destroy_user_lock_with_checks( kmp_user_lock_p lck )
{
    KMP_DEBUG_ASSERT( __kmp_destroy_user_lock_with_checks_ != NULL );
    ( *__kmp_destroy_user_lock_with_checks_ )( lck );
}

extern void ( *__kmp_acquire_nested_user_lock_with_checks_ )( kmp_user_lock_p lck, kmp_int32 gtid );

#if KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64)

#define __kmp_acquire_nested_user_lock_with_checks(lck,gtid)                                        \
    if (__kmp_user_lock_kind == lk_tas) {                                                           \
        if ( __kmp_env_consistency_check ) {                                                        \
            char const * const func = "omp_set_nest_lock";                                          \
            if ( ( sizeof ( kmp_tas_lock_t ) <= OMP_NEST_LOCK_T_SIZE )                              \
                && lck->tas.lk.depth_locked == -1 ) {                                               \
                KMP_FATAL( LockSimpleUsedAsNestable, func );                                        \
            }                                                                                       \
        }                                                                                           \
        if ( lck->tas.lk.poll - 1 == gtid ) {                                                       \
            lck->tas.lk.depth_locked += 1;                                                          \
        } else {                                                                                    \
            if ( ( lck->tas.lk.poll != 0 ) ||                                                       \
              ( ! KMP_COMPARE_AND_STORE_ACQ32( &(lck->tas.lk.poll), 0, gtid + 1 ) )  ) {            \
                kmp_uint32 spins;                                                                   \
                KMP_FSYNC_PREPARE( lck );                                                           \
                KMP_INIT_YIELD( spins );                                                            \
                if ( TCR_4(__kmp_nth) > (__kmp_avail_proc ? __kmp_avail_proc : __kmp_xproc) ) {     \
                    KMP_YIELD( TRUE );                                                              \
                } else {                                                                            \
                    KMP_YIELD_SPIN( spins );                                                        \
                }                                                                                   \
                while ( ( lck->tas.lk.poll != 0 ) ||                                                \
                  ( ! KMP_COMPARE_AND_STORE_ACQ32( &(lck->tas.lk.poll), 0, gtid + 1 ) )  ) {        \
                    if ( TCR_4(__kmp_nth) > (__kmp_avail_proc ? __kmp_avail_proc : __kmp_xproc) ) { \
                        KMP_YIELD( TRUE );                                                          \
                    } else {                                                                        \
                        KMP_YIELD_SPIN( spins );                                                    \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
            lck->tas.lk.depth_locked = 1;                                                           \
        }                                                                                           \
        KMP_FSYNC_ACQUIRED( lck );                                                                  \
    } else {                                                                                        \
        KMP_DEBUG_ASSERT( __kmp_acquire_nested_user_lock_with_checks_ != NULL );                    \
        ( *__kmp_acquire_nested_user_lock_with_checks_ )( lck, gtid );                              \
    }

#else
static inline void
__kmp_acquire_nested_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    KMP_DEBUG_ASSERT( __kmp_acquire_nested_user_lock_with_checks_ != NULL );
    ( *__kmp_acquire_nested_user_lock_with_checks_ )( lck, gtid );
}
#endif

extern int ( *__kmp_test_nested_user_lock_with_checks_ )( kmp_user_lock_p lck, kmp_int32 gtid );

#if KMP_OS_LINUX && (KMP_ARCH_X86 || KMP_ARCH_X86_64)
static inline int
__kmp_test_nested_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    if ( __kmp_user_lock_kind == lk_tas ) {
        int retval;
        if ( __kmp_env_consistency_check ) {
            char const * const func = "omp_test_nest_lock";
            if ( ( sizeof ( kmp_tas_lock_t ) <= OMP_NEST_LOCK_T_SIZE )
                && lck->tas.lk.depth_locked == -1 ) {
                KMP_FATAL( LockSimpleUsedAsNestable, func );
            }
        }
        KMP_DEBUG_ASSERT( gtid >= 0 );
        if ( lck->tas.lk.poll - 1 == gtid ) {   /* __kmp_get_tas_lock_owner( lck ) == gtid */
            return ++lck->tas.lk.depth_locked;  /* same owner, depth increased */
        }
        retval = ( ( lck->tas.lk.poll == 0 ) &&
          KMP_COMPARE_AND_STORE_ACQ32( &(lck->tas.lk.poll), 0, gtid + 1 ) );
        if ( retval ) {
            KMP_MB();
            lck->tas.lk.depth_locked = 1;
        }
        return retval;
    } else {
        KMP_DEBUG_ASSERT( __kmp_test_nested_user_lock_with_checks_ != NULL );
        return ( *__kmp_test_nested_user_lock_with_checks_ )( lck, gtid );
    }
}
#else
static inline int
__kmp_test_nested_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    KMP_DEBUG_ASSERT( __kmp_test_nested_user_lock_with_checks_ != NULL );
    return ( *__kmp_test_nested_user_lock_with_checks_ )( lck, gtid );
}
#endif

extern void ( *__kmp_release_nested_user_lock_with_checks_ )( kmp_user_lock_p lck, kmp_int32 gtid );

static inline void
__kmp_release_nested_user_lock_with_checks( kmp_user_lock_p lck, kmp_int32 gtid )
{
    KMP_DEBUG_ASSERT( __kmp_release_nested_user_lock_with_checks_ != NULL );
    ( *__kmp_release_nested_user_lock_with_checks_ )( lck, gtid );
}

extern void ( *__kmp_init_nested_user_lock_with_checks_ )( kmp_user_lock_p lck );

static inline void __kmp_init_nested_user_lock_with_checks( kmp_user_lock_p lck )
{
    KMP_DEBUG_ASSERT( __kmp_init_nested_user_lock_with_checks_ != NULL );
    ( *__kmp_init_nested_user_lock_with_checks_ )( lck );
}

extern void ( *__kmp_destroy_nested_user_lock_with_checks_ )( kmp_user_lock_p lck );

static inline void
__kmp_destroy_nested_user_lock_with_checks( kmp_user_lock_p lck )
{
    KMP_DEBUG_ASSERT( __kmp_destroy_nested_user_lock_with_checks_ != NULL );
    ( *__kmp_destroy_nested_user_lock_with_checks_ )( lck );
}

//
// user lock functions which do not necessarily exist for all lock kinds.
//
// The "set" functions usually have wrapper routines that check for a NULL set
// function pointer and call it if non-NULL.
//
// In some cases, it makes sense to have a "get" wrapper function check for a
// NULL get function pointer and return NULL / invalid value / error code if
// the function pointer is NULL.
//
// In other cases, the calling code really should differentiate between an
// unimplemented function and one that is implemented but returning NULL /
// invalied value.  If this is the case, no get function wrapper exists.
//

extern int ( *__kmp_is_user_lock_initialized_ )( kmp_user_lock_p lck );

// no set function; fields set durining local allocation

extern const ident_t * ( *__kmp_get_user_lock_location_ )( kmp_user_lock_p lck );

static inline const ident_t *
__kmp_get_user_lock_location( kmp_user_lock_p lck )
{
    if ( __kmp_get_user_lock_location_  != NULL ) {
        return ( *__kmp_get_user_lock_location_ )( lck );
    }
    else {
        return NULL;
    }
}

extern void ( *__kmp_set_user_lock_location_ )( kmp_user_lock_p lck, const ident_t *loc );

static inline void
__kmp_set_user_lock_location( kmp_user_lock_p lck, const ident_t *loc )
{
    if ( __kmp_set_user_lock_location_  != NULL ) {
        ( *__kmp_set_user_lock_location_ )( lck, loc );
    }
}

extern kmp_lock_flags_t ( *__kmp_get_user_lock_flags_ )( kmp_user_lock_p lck );

extern void ( *__kmp_set_user_lock_flags_ )( kmp_user_lock_p lck, kmp_lock_flags_t flags );

static inline void
__kmp_set_user_lock_flags( kmp_user_lock_p lck, kmp_lock_flags_t flags )
{
    if ( __kmp_set_user_lock_flags_  != NULL ) {
        ( *__kmp_set_user_lock_flags_ )( lck, flags );
    }
}

//
// The fuction which sets up all of the vtbl pointers for kmp_user_lock_t.
//
extern void __kmp_set_user_lock_vptrs( kmp_lock_kind_t user_lock_kind );



// ----------------------------------------------------------------------------
// User lock table & lock allocation
// ----------------------------------------------------------------------------

/*
    On 64-bit Linux* OS (and OS X*) GNU compiler allocates only 4 bytems memory for lock variable, which
    is not enough to store a pointer, so we have to use lock indexes instead of pointers and
    maintain lock table to map indexes to pointers.


    Note: The first element of the table is not a pointer to lock! It is a pointer to previously
    allocated table (or NULL if it is the first table).

    Usage:

        if ( OMP_LOCK_T_SIZE < sizeof( <lock> ) ) { // or OMP_NEST_LOCK_T_SIZE
            Lock table is fully utilized. User locks are indexes, so table is
            used on user lock operation.
            Note: it may be the case (lin_32) that we don't need to use a lock
            table for regular locks, but do need the table for nested locks.
        }
        else {
            Lock table initialized but not actually used.
        }
*/

struct kmp_lock_table {
    kmp_lock_index_t  used;      // Number of used elements
    kmp_lock_index_t  allocated; // Number of allocated elements
    kmp_user_lock_p * table;     // Lock table.
};

typedef struct kmp_lock_table kmp_lock_table_t;

extern kmp_lock_table_t __kmp_user_lock_table;
extern kmp_user_lock_p __kmp_lock_pool;

struct kmp_block_of_locks {
    struct kmp_block_of_locks * next_block;
    void *                      locks;
};

typedef struct kmp_block_of_locks kmp_block_of_locks_t;

extern kmp_block_of_locks_t *__kmp_lock_blocks;
extern int __kmp_num_locks_in_block;

extern kmp_user_lock_p __kmp_user_lock_allocate( void **user_lock, kmp_int32 gtid, kmp_lock_flags_t flags );
extern void __kmp_user_lock_free( void **user_lock, kmp_int32 gtid, kmp_user_lock_p lck );
extern kmp_user_lock_p __kmp_lookup_user_lock( void **user_lock, char const *func );
extern void __kmp_cleanup_user_locks();

#define KMP_CHECK_USER_LOCK_INIT() \
        {                                                               \
            if ( ! TCR_4( __kmp_init_user_locks ) ) {                   \
                __kmp_acquire_bootstrap_lock( &__kmp_initz_lock );      \
                if ( ! TCR_4( __kmp_init_user_locks ) ) {               \
                    TCW_4( __kmp_init_user_locks, TRUE );               \
                }                                                       \
                __kmp_release_bootstrap_lock( &__kmp_initz_lock );      \
            }                                                           \
        }

#undef KMP_PAD
#undef KMP_GTID_DNE

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* KMP_LOCK_H */

