// A five-level pointer chain, p1 -> p2 -> p3 -> p4 -> p5 -> x, where the
// intermediate levels are all mapped and attached on one inner construct while
// the ends may already be mapped from earlier ones.
//
// Under unified shared memory a mapping's corresponding storage is normally the
// original storage, so attaching a pointer would assign a device address to the
// original pointer. One side has to hold device storage instead, and only a
// mapping created by the current construct may change: the device address of one
// that was already present may already have been obtained by the program.
//
// The chain makes the two directions visible, because a decision at one level
// propagates to its neighbours:
//
//  - giving a pointee's allocation up (demotion) removes the disparity at that
//    level outright, and needs nothing from the rest of the chain;
//  - giving a pointer device storage (upgrade) moves that pointer's own address,
//    so whatever points at it must be able to hold a device address too, which
//    walks backwards towards the root -- including through attachments this
//    construct has not performed yet.
//
// So a device-backed leaf drags the whole chain to device storage, while a
// device-backed root needs nothing of the levels below it. When both ends are
// already present and pull in opposite directions, neither can move and the
// runtime reports it rather than assigning a device address to an original
// pointer.
//
// The scenarios below cover each combination of which end pre-exists and how,
// and then the same conflicts originating from a level in the middle.
//
// RUN: %libomptarget-compile-generic -DS=1
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic -check-prefixes=S1
//
// RUN: %libomptarget-compile-generic -DS=2
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefixes=S2,OK
//
// RUN: %libomptarget-compile-generic -DS=3
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefixes=S3,OK
//
// RUN: %libomptarget-compile-generic -DS=4
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefixes=S4,OK
//
// RUN: %libomptarget-compile-generic -DS=5
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefixes=S5,OK
//
// RUN: %libomptarget-compile-generic -DS=6
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefixes=S6,OK
//
// RUN: %libomptarget-compile-generic -DS=7
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefixes=S7,OK
//
// RUN: %libomptarget-compile-generic -DS=8
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefixes=S8,OK
//
// RUN: %libomptarget-compile-generic -DS=9
// RUN: %libomptarget-run-fail-generic 2>&1 | %fcheck-generic -check-prefixes=S9
//
// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9
//
// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

#include <stdio.h>

#pragma omp requires unified_shared_memory

// p1 -> p2 -> p3 -> p4 -> p5 -> x
int x[10];
int *p5 = &x[0];
int **p4 = &p5;
int ***p3 = &p4;
int ****p2 = &p3;
int *****p1 = &p2;

void *dev_addr[5];
void *dev_val[5];
void *dev_chain_leaf;
int dev_read;

int main(void) {
  for (int i = 0; i < 10; ++i)
    x[i] = 42;

#pragma omp target
  {
  }

  // ---- pre-existing mappings ----
#if S == 1
  // x device-backed (close), p1 host-backed. Both already present.
#pragma omp target enter data map(close, to : x[0 : 10])
#pragma omp target enter data map(alloc : p1)
#elif S == 2
  // reverse: p1 device-backed (close), x host-backed.
#pragma omp target enter data map(close, alloc : p1)
#pragma omp target enter data map(to : x[0 : 10])
#elif S == 3
  // only the leaf x exists, device-backed.
#pragma omp target enter data map(close, to : x[0 : 10])
#elif S == 4
  // only the root p1 exists, host-backed.
#pragma omp target enter data map(alloc : p1)
#elif S == 5
  // only the root p1 exists, device-backed.
#pragma omp target enter data map(close, alloc : p1)
#elif S == 6
  // both ends exist and are device-backed.
#pragma omp target enter data map(close, to : x[0 : 10])
#pragma omp target enter data map(close, alloc : p1)
#elif S == 7
  // a middle level pre-exists, device-backed: p3's storage holds the pointer p3,
  // so this is the pointer that p2 will be attached to.
#pragma omp target enter data map(close, alloc : p3)
#elif S == 8
  // a middle level pre-exists sharing storage with the original.
#pragma omp target enter data map(alloc : p3)
#elif S == 9
  // the leaf is device-backed and a middle level pre-exists sharing storage with
  // the original, so the upgrade walking up from the leaf runs into it.
#pragma omp target enter data map(close, to : x[0 : 10])
#pragma omp target enter data map(alloc : p3)
#endif

  // ---- inner: all intermediate levels mapped + all attachments ----
#if S == 4 || S == 5 || S == 7 || S == 8
  // x is not pre-mapped here, so close on the leaf makes it newly device-backed.
#pragma omp target data map(alloc : p1)                                        \
    map(alloc : p1[0 : 1], p2[0 : 1], p3[0 : 1], p4[0 : 1])                   \
    map(close, to : p5[0 : 10])
#else
#pragma omp target data map(alloc : p1)                                        \
    map(alloc : p1[0 : 1], p2[0 : 1], p3[0 : 1], p4[0 : 1])                   \
    map(alloc : p5[0 : 10])
#endif
  {
#pragma omp target map(present, alloc : p1, p2, p3, p4, p5)                    \
    map(from : dev_addr, dev_val, dev_chain_leaf, dev_read)
    {
      dev_addr[0] = (void *)&p1;
      dev_addr[1] = (void *)&p2;
      dev_addr[2] = (void *)&p3;
      dev_addr[3] = (void *)&p4;
      dev_addr[4] = (void *)&p5;
      dev_val[0] = (void *)p1;
      dev_val[1] = (void *)p2;
      dev_val[2] = (void *)p3;
      dev_val[3] = (void *)p4;
      dev_val[4] = (void *)p5;
      // reach the leaf through the whole chain
      dev_chain_leaf = (void *)(****p1);
      dev_read = *****p1;
      *****p1 = 777;
    }
  }

  void *host_addr[5] = {(void *)&p1, (void *)&p2, (void *)&p3, (void *)&p4,
                        (void *)&p5};
  void *want_val[5] = {(void *)&p2, (void *)&p3, (void *)&p4, (void *)&p5,
                       (void *)&x[0]};
  void *have_val[5] = {(void *)p1, (void *)p2, (void *)p3, (void *)p4,
                       (void *)p5};
  const char *nm[5] = {"p1", "p2", "p3", "p4", "p5"};

  printf("scenario %d\n", S);
  for (int i = 0; i < 5; ++i)
    printf("  %s: storage=%-6s devval=%-9s host_restored=%s\n", nm[i],
           dev_addr[i] == host_addr[i] ? "shared" : "device",
           dev_val[i] == want_val[i] ? "host-addr" : "device-addr",
           have_val[i] == want_val[i] ? "yes" : "NO");
  printf("  chain: %s\n",
         dev_chain_leaf == dev_val[4] ? "consistent" : "STRANDED");
  printf("  dev_read=%d (want 42)  x[0]=%d (want 777)\n", dev_read, x[0]);

  // Both ends are already present and pull in opposite directions: the leaf x is
  // device-backed so it cannot give its allocation up, which forces every level
  // up to p1 onto device storage, but p1 is already present sharing storage with
  // the original and so cannot take an allocation now.
  //
  // clang-format off
  // S1: could not do pointer attachment
  // S1-SAME: would have to be device-bound as well
  // clang-format on

  // p1 is already device-backed, so the attachment into it does not touch an
  // original pointer, and x shares storage with the original, so no level below
  // needs to change.
  // S2: scenario 2
  // S2: p1: storage=device devval=host-addr host_restored=yes
  // S2: p2: storage=shared devval=host-addr host_restored=yes
  // S2: p3: storage=shared devval=host-addr host_restored=yes
  // S2: p4: storage=shared devval=host-addr host_restored=yes
  // S2: p5: storage=shared devval=host-addr host_restored=yes

  // Only the leaf pre-exists, device-backed. It cannot be demoted, so p5 must
  // take device storage, and that walks backwards through the pending
  // attachments to every level up to p1 -- all of which this construct maps, so
  // all of them can.
  // S3: scenario 3
  // S3: p1: storage=device devval=device-addr host_restored=yes
  // S3: p2: storage=device devval=device-addr host_restored=yes
  // S3: p3: storage=device devval=device-addr host_restored=yes
  // S3: p4: storage=device devval=device-addr host_restored=yes
  // S3: p5: storage=device devval=device-addr host_restored=yes

  // Only the root pre-exists, sharing storage with the original. The leaf's
  // allocation is made by this construct, so it is given up instead, and the
  // whole chain stays on the host path.
  // S4: scenario 4
  // S4: p1: storage=shared devval=host-addr host_restored=yes
  // S4: p2: storage=shared devval=host-addr host_restored=yes
  // S4: p3: storage=shared devval=host-addr host_restored=yes
  // S4: p4: storage=shared devval=host-addr host_restored=yes
  // S4: p5: storage=shared devval=host-addr host_restored=yes

  // Only the root pre-exists, device-backed. Demoting the leaf still settles
  // every level below, and p1's own storage is left as it was.
  // S5: scenario 5
  // S5: p1: storage=device devval=host-addr host_restored=yes
  // S5: p2: storage=shared devval=host-addr host_restored=yes
  // S5: p3: storage=shared devval=host-addr host_restored=yes
  // S5: p4: storage=shared devval=host-addr host_restored=yes
  // S5: p5: storage=shared devval=host-addr host_restored=yes

  // Both ends pre-exist device-backed, so they agree: the leaf drags the chain
  // onto device storage and p1 was already there.
  // S6: scenario 6
  // S6: p1: storage=device devval=device-addr host_restored=yes
  // S6: p2: storage=device devval=device-addr host_restored=yes
  // S6: p3: storage=device devval=device-addr host_restored=yes
  // S6: p4: storage=device devval=device-addr host_restored=yes
  // S6: p5: storage=device devval=device-addr host_restored=yes

  // A device-backed level in the middle settles everything below it, and drags
  // only the levels above it: p2 is attached to p3's storage, which is already
  // device-backed, so p2 -- and then p1 -- must take device storage, while p4 and
  // p5 never see a disparity and stay put. The cascade stops where the chain
  // stops needing it.
  // S7: scenario 7
  // S7: p1: storage=device devval=device-addr host_restored=yes
  // S7: p2: storage=device devval=device-addr host_restored=yes
  // S7: p3: storage=device devval=host-addr host_restored=yes
  // S7: p4: storage=shared devval=host-addr host_restored=yes
  // S7: p5: storage=shared devval=host-addr host_restored=yes

  // A middle level sharing storage with the original conflicts with nothing: the
  // leaf's allocation is given up and the whole chain stays on the host path.
  // S8: scenario 8
  // S8: p1: storage=shared devval=host-addr host_restored=yes
  // S8: p2: storage=shared devval=host-addr host_restored=yes
  // S8: p3: storage=shared devval=host-addr host_restored=yes
  // S8: p4: storage=shared devval=host-addr host_restored=yes
  // S8: p5: storage=shared devval=host-addr host_restored=yes

  // The blocker need not be at the root. Here the device-backed leaf forces the
  // upgrade to walk up the chain, and it reaches p3, which was already mapped
  // sharing storage with the original and so cannot take an allocation now.
  //
  // clang-format off
  // S9: could not do pointer attachment
  // S9-SAME: would have to be device-bound as well
  // clang-format on

  // Wherever the chain ends up, it must be internally consistent: reaching the
  // leaf through all five levels must arrive at the same storage p5 designates,
  // and every original pointer must be intact afterwards.
  // OK: chain: consistent

  // The kernel's write lands in whatever storage the leaf ended up in. Where the
  // chain was dragged onto device storage that is the device buffer, and neither
  // alloc nor to copies it back, so the original x is unchanged.
  // S2: dev_read=42 (want 42)  x[0]=777 (want 777)
  // S3: dev_read=42 (want 42)  x[0]=42 (want 777)
  // S4: dev_read=42 (want 42)  x[0]=777 (want 777)
  // S5: dev_read=42 (want 42)  x[0]=777 (want 777)
  // S6: dev_read=42 (want 42)  x[0]=42 (want 777)
  // S7: dev_read=42 (want 42)  x[0]=777 (want 777)
  // S8: dev_read=42 (want 42)  x[0]=777 (want 777)
  return 0;
}
