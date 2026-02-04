# Test that -j1 runs in single-process mode (no multiprocessing).
#
# This ensures lit can run in sandboxed environments where multiprocessing
# primitives (semaphores, process creation) are blocked.

# REQUIRES: strace

# Use strace to trace file operations. With -j1, lit should NOT create
# POSIX semaphores in /dev/shm which are used by multiprocessing.Pool.
#
# Python's multiprocessing.BoundedSemaphore creates files like:
#   openat(AT_FDCWD, "/dev/shm/sem.XXXXXX", O_RDWR|O_CREAT|O_EXCL|...)
#
# These are the semaphores that fail in sandboxed environments, so verifying
# they don't appear with -j1 confirms the fix works.

# RUN: export PATH="%{system-path}"
#
# RUN: strace -f -e trace=openat not %{lit} -j1 %{inputs}/shtest-format 2>&1 \
# RUN:   | FileCheck --check-prefix=CHECK-J1 %s
#
# CHECK-J1: 1 workers
# CHECK-J1-NOT: /dev/shm/sem
# CHECK-J1: Total Discovered Tests:

# With -j2 and multiple tests, lit MUST use multiprocessing.Pool which creates
# POSIX semaphores. This validates that the CHECK-NOT pattern above is correct
# and won't become stale if Python/glibc changes how semaphores are created.
#
# RUN: strace -f -e trace=openat not %{lit} -j2 %{inputs}/shtest-format 2>&1 \
# RUN:   | FileCheck --check-prefix=CHECK-J2 %s
#
# CHECK-J2: /dev/shm/sem
# CHECK-J2: 2 workers
# CHECK-J2: Total Discovered Tests:
