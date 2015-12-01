/*
 * kmp_affinity.h -- header for affinity management
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#ifndef KMP_AFFINITY_H
#define KMP_AFFINITY_H

extern int __kmp_affinity_compact; /* Affinity 'compact' value */

class Address {
public:
    static const unsigned maxDepth = 32;
    unsigned labels[maxDepth];
    unsigned childNums[maxDepth];
    unsigned depth;
    unsigned leader;
    Address(unsigned _depth)
      : depth(_depth), leader(FALSE) {
    }
    Address &operator=(const Address &b) {
        depth = b.depth;
        for (unsigned i = 0; i < depth; i++) {
            labels[i] = b.labels[i];
            childNums[i] = b.childNums[i];
        }
        leader = FALSE;
        return *this;
    }
    bool operator==(const Address &b) const {
        if (depth != b.depth)
            return false;
        for (unsigned i = 0; i < depth; i++)
            if(labels[i] != b.labels[i])
                return false;
        return true;
    }
    bool isClose(const Address &b, int level) const {
        if (depth != b.depth)
            return false;
        if ((unsigned)level >= depth)
            return true;
        for (unsigned i = 0; i < (depth - level); i++)
            if(labels[i] != b.labels[i])
                return false;
        return true;
    }
    bool operator!=(const Address &b) const {
        return !operator==(b);
    }
    void print() const {
        unsigned i;
        printf("Depth: %u --- ", depth);
        for(i=0;i<depth;i++) {
            printf("%u ", labels[i]);
        }
    }
};

class AddrUnsPair {
public:
    Address first;
    unsigned second;
    AddrUnsPair(Address _first, unsigned _second)
      : first(_first), second(_second) {
    }
    AddrUnsPair &operator=(const AddrUnsPair &b)
    {
        first = b.first;
        second = b.second;
        return *this;
    }
    void print() const {
        printf("first = "); first.print();
        printf(" --- second = %u", second);
    }
    bool operator==(const AddrUnsPair &b) const {
        if(first != b.first) return false;
        if(second != b.second) return false;
        return true;
    }
    bool operator!=(const AddrUnsPair &b) const {
        return !operator==(b);
    }
};


static int
__kmp_affinity_cmp_Address_labels(const void *a, const void *b)
{
    const Address *aa = (const Address *)&(((AddrUnsPair *)a)
      ->first);
    const Address *bb = (const Address *)&(((AddrUnsPair *)b)
      ->first);
    unsigned depth = aa->depth;
    unsigned i;
    KMP_DEBUG_ASSERT(depth == bb->depth);
    for (i  = 0; i < depth; i++) {
        if (aa->labels[i] < bb->labels[i]) return -1;
        if (aa->labels[i] > bb->labels[i]) return 1;
    }
    return 0;
}


static int
__kmp_affinity_cmp_Address_child_num(const void *a, const void *b)
{
    const Address *aa = (const Address *)&(((AddrUnsPair *)a)
      ->first);
    const Address *bb = (const Address *)&(((AddrUnsPair *)b)
      ->first);
    unsigned depth = aa->depth;
    unsigned i;
    KMP_DEBUG_ASSERT(depth == bb->depth);
    KMP_DEBUG_ASSERT((unsigned)__kmp_affinity_compact <= depth);
    KMP_DEBUG_ASSERT(__kmp_affinity_compact >= 0);
    for (i = 0; i < (unsigned)__kmp_affinity_compact; i++) {
        int j = depth - i - 1;
        if (aa->childNums[j] < bb->childNums[j]) return -1;
        if (aa->childNums[j] > bb->childNums[j]) return 1;
    }
    for (; i < depth; i++) {
        int j = i - __kmp_affinity_compact;
        if (aa->childNums[j] < bb->childNums[j]) return -1;
        if (aa->childNums[j] > bb->childNums[j]) return 1;
    }
    return 0;
}


/** A structure for holding machine-specific hierarchy info to be computed once at init.
    This structure represents a mapping of threads to the actual machine hierarchy, or to
    our best guess at what the hierarchy might be, for the purpose of performing an
    efficient barrier.  In the worst case, when there is no machine hierarchy information,
    it produces a tree suitable for a barrier, similar to the tree used in the hyper barrier. */
class hierarchy_info {
public:
    /** Good default values for number of leaves and branching factor, given no affinity information.
	Behaves a bit like hyper barrier. */
    static const kmp_uint32 maxLeaves=4;
    static const kmp_uint32 minBranch=4;
    /** Number of levels in the hierarchy. Typical levels are threads/core, cores/package
	or socket, packages/node, nodes/machine, etc.  We don't want to get specific with
	nomenclature.  When the machine is oversubscribed we add levels to duplicate the
	hierarchy, doubling the thread capacity of the hierarchy each time we add a level. */
    kmp_uint32 maxLevels;

    /** This is specifically the depth of the machine configuration hierarchy, in terms of the
        number of levels along the longest path from root to any leaf. It corresponds to the
        number of entries in numPerLevel if we exclude all but one trailing 1. */
    kmp_uint32 depth;
    kmp_uint32 base_num_threads;
    enum init_status { initialized=0, not_initialized=1, initializing=2 };
    volatile kmp_int8 uninitialized; // 0=initialized, 1=not initialized, 2=initialization in progress
    volatile kmp_int8 resizing; // 0=not resizing, 1=resizing

    /** Level 0 corresponds to leaves. numPerLevel[i] is the number of children the parent of a
        node at level i has. For example, if we have a machine with 4 packages, 4 cores/package
        and 2 HT per core, then numPerLevel = {2, 4, 4, 1, 1}. All empty levels are set to 1. */
    kmp_uint32 *numPerLevel;
    kmp_uint32 *skipPerLevel;

    void deriveLevels(AddrUnsPair *adr2os, int num_addrs) {
        int hier_depth = adr2os[0].first.depth;
        int level = 0;
        for (int i=hier_depth-1; i>=0; --i) {
            int max = -1;
            for (int j=0; j<num_addrs; ++j) {
                int next = adr2os[j].first.childNums[i];
                if (next > max) max = next;
            }
            numPerLevel[level] = max+1;
            ++level;
        }
    }

    hierarchy_info() : maxLevels(7), depth(1), uninitialized(not_initialized), resizing(0) {}

    void fini() { if (!uninitialized && numPerLevel) __kmp_free(numPerLevel); }

    void init(AddrUnsPair *adr2os, int num_addrs)
    {
        kmp_int8 bool_result = KMP_COMPARE_AND_STORE_ACQ8(&uninitialized, not_initialized, initializing);
        if (bool_result == 0) { // Wait for initialization
            while (TCR_1(uninitialized) != initialized) KMP_CPU_PAUSE();
            return;
        }
        KMP_DEBUG_ASSERT(bool_result==1);

        /* Added explicit initialization of the data fields here to prevent usage of dirty value
           observed when static library is re-initialized multiple times (e.g. when
           non-OpenMP thread repeatedly launches/joins thread that uses OpenMP). */
        depth = 1;
        resizing = 0;
        maxLevels = 7;
        numPerLevel = (kmp_uint32 *)__kmp_allocate(maxLevels*2*sizeof(kmp_uint32));
        skipPerLevel = &(numPerLevel[maxLevels]);
        for (kmp_uint32 i=0; i<maxLevels; ++i) { // init numPerLevel[*] to 1 item per level
            numPerLevel[i] = 1;
            skipPerLevel[i] = 1;
        }

        // Sort table by physical ID
        if (adr2os) {
            qsort(adr2os, num_addrs, sizeof(*adr2os), __kmp_affinity_cmp_Address_labels);
            deriveLevels(adr2os, num_addrs);
        }
        else {
            numPerLevel[0] = maxLeaves;
            numPerLevel[1] = num_addrs/maxLeaves;
            if (num_addrs%maxLeaves) numPerLevel[1]++;
        }

        base_num_threads = num_addrs;
        for (int i=maxLevels-1; i>=0; --i) // count non-empty levels to get depth
            if (numPerLevel[i] != 1 || depth > 1) // only count one top-level '1'
                depth++;

        kmp_uint32 branch = minBranch;
        if (numPerLevel[0] == 1) branch = num_addrs/maxLeaves;
        if (branch<minBranch) branch=minBranch;
        for (kmp_uint32 d=0; d<depth-1; ++d) { // optimize hierarchy width
            while (numPerLevel[d] > branch || (d==0 && numPerLevel[d]>maxLeaves)) { // max 4 on level 0!
                if (numPerLevel[d] & 1) numPerLevel[d]++;
                numPerLevel[d] = numPerLevel[d] >> 1;
                if (numPerLevel[d+1] == 1) depth++;
                numPerLevel[d+1] = numPerLevel[d+1] << 1;
            }
            if(numPerLevel[0] == 1) {
                branch = branch >> 1;
                if (branch<4) branch = minBranch;
            }
        }

        for (kmp_uint32 i=1; i<depth; ++i)
            skipPerLevel[i] = numPerLevel[i-1] * skipPerLevel[i-1];
        // Fill in hierarchy in the case of oversubscription
        for (kmp_uint32 i=depth; i<maxLevels; ++i)
            skipPerLevel[i] = 2*skipPerLevel[i-1];

        uninitialized = initialized; // One writer

    }

    // Resize the hierarchy if nproc changes to something larger than before
    void resize(kmp_uint32 nproc)
    {
        kmp_int8 bool_result = KMP_COMPARE_AND_STORE_ACQ8(&resizing, 0, 1);
        while (bool_result == 0) { // someone else is trying to resize
            KMP_CPU_PAUSE();
            if (nproc <= base_num_threads)  // happy with other thread's resize
                return;
            else // try to resize
                bool_result = KMP_COMPARE_AND_STORE_ACQ8(&resizing, 0, 1);
        }
        KMP_DEBUG_ASSERT(bool_result!=0);
        if (nproc <= base_num_threads) return; // happy with other thread's resize

        // Calculate new maxLevels
        kmp_uint32 old_sz = skipPerLevel[depth-1];
        kmp_uint32 incs = 0, old_maxLevels = maxLevels;
        // First see if old maxLevels is enough to contain new size
        for (kmp_uint32 i=depth; i<maxLevels && nproc>old_sz; ++i) {
            skipPerLevel[i] = 2*skipPerLevel[i-1];
            numPerLevel[i-1] *= 2;
            old_sz *= 2;
            depth++;
        }
        if (nproc > old_sz) { // Not enough space, need to expand hierarchy
            while (nproc > old_sz) {
                old_sz *=2;
                incs++;
                depth++;
            }
            maxLevels += incs;

            // Resize arrays
            kmp_uint32 *old_numPerLevel = numPerLevel;
            kmp_uint32 *old_skipPerLevel = skipPerLevel;
            numPerLevel = skipPerLevel = NULL;
            numPerLevel = (kmp_uint32 *)__kmp_allocate(maxLevels*2*sizeof(kmp_uint32));
            skipPerLevel = &(numPerLevel[maxLevels]);

            // Copy old elements from old arrays
            for (kmp_uint32 i=0; i<old_maxLevels; ++i) { // init numPerLevel[*] to 1 item per level
                numPerLevel[i] = old_numPerLevel[i];
                skipPerLevel[i] = old_skipPerLevel[i];
            }

            // Init new elements in arrays to 1
            for (kmp_uint32 i=old_maxLevels; i<maxLevels; ++i) { // init numPerLevel[*] to 1 item per level
                numPerLevel[i] = 1;
                skipPerLevel[i] = 1;
            }

            // Free old arrays
            __kmp_free(old_numPerLevel);
        }

        // Fill in oversubscription levels of hierarchy
        for (kmp_uint32 i=old_maxLevels; i<maxLevels; ++i)
            skipPerLevel[i] = 2*skipPerLevel[i-1];

        base_num_threads = nproc;
        resizing = 0; // One writer

    }
};
#endif // KMP_AFFINITY_H
