How to create vmcores for tests
===============================

1. Boot a FreeBSD VM with as little memory as possible and create a core dump
   per `FreeBSD Handbook Kernel Debugging Chapter`_.  Note that you may need to
   reboot with more memory after the kernel panic as otherwise savecore(8) may
   fail.

   For instance, I was able to boot FreeBSD and qemu-system-x86_64 with 128 MiB
   RAM but had to increase it to 256 MiB for the boot after kernel panic.

2. Transfer the kernel image (``/boot/kernel/kernel``) and vmcore
   (``/var/crash/vmcore.latest``) from the VM.

3. Patch libfbsdvmcore using ``libfbsdvmcore-hacks.patch`` and build LLDB
   against the patched library.

4. Patch LLDB using ``lldb-minimize-processes.patch`` and build it.

   WARNING: LLDB will now modify core files in order to make the resulting
   test vmcores smaller.  Make a backup if necessary.

5. Do a test run of ``test.script`` in LLDB against the kernel + vmcore::

    lldb -b -s test.script --core /path/to/core /path/to/kernel

   If everything works fine, the LLDB output should be interspersed with
   ``%RD`` lines.  The vmcore will also be modified to shorten the process
   list in ``allproc``.

6. Open the vmcore in the patched LLDB again and choose interesting threads
   for testing.  Update thread numbers in ``test.script`` if necessary.

7. Use the ``copy-sparse.py`` tool to create a sparse version of the vmcore::

       lldb -b -s test.script --core /path/to/core /path/to/kernel |
           grep '^% RD' | python copy-sparse.py /path/to/core vmcore.sparse

8. Compress the sparse vmcore file using ``bzip2``::

       bzip2 -9 vmcore.sparse


.. _FreeBSD Handbook Kernel Debugging Chapter:
   https://docs.freebsd.org/en/books/developers-handbook/kerneldebug/
