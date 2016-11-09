====================
XRay Instrumentation
====================

:Version: 1 as of 2016-11-08

.. contents::
   :local:


Introduction
============

XRay is a function call tracing system which combines compiler-inserted
instrumentation points and a runtime library that can dynamically enable and
disable the instrumentation.

More high level information about XRay can be found in the `XRay whitepaper`_.

This document describes how to use XRay as implemented in LLVM.

XRay in LLVM
============

XRay consists of three main parts:

- Compiler-inserted instrumentation points.
- A runtime library for enabling/disabling tracing at runtime.
- A suite of tools for analysing the traces.

  **NOTE:** As of the time of this writing, XRay is only available for x86_64
  and arm7 32-bit (no-thumb) Linux.

The compiler-inserted instrumentation points come in the form of nop-sleds in
the final generated binary, and an ELF section named ``xray_instr_map`` which
contains entries pointing to these instrumentation points. The runtime library
relies on being able to access the entries of the ``xray_instr_map``, and
overwrite the instrumentation points at runtime.

Using XRay
==========

You can use XRay in a couple of ways:

- Instrumenting your C/C++/Objective-C/Objective-C++ application.
- Generating LLVM IR with the correct function attributes.

The rest of this section covers these main ways and later on how to customise
what XRay does in an XRay-instrumented binary.

Instrumenting your C/C++/Objective-C Application
------------------------------------------------

The easiest way of getting XRay instrumentation for your application is by
enabling the ``-fxray-instrument`` flag in your clang invocation.

For example:

::

  clang -fxray-instrument ..

By default, functions that have at least 200 instructions will get XRay
instrumentation points. You can tweak that number through the
``-fxray-instruction-threshold=`` flag:

::

  clang -fxray-instrument -fxray-instruction-threshold=1 ..

You can also specifically instrument functions in your binary to either always
or never be instrumented using source-level attributes. You can do it using the
GCC-style attributes or C++11-style attributes.

.. code-block:: c++

    [[clang::xray_always_intrument]] void always_instrumented();

    [[clang::xray_never_instrument]] void never_instrumented();

    void alt_always_instrumented() __attribute__((xray_always_intrument));

    void alt_never_instrumented() __attribute__((xray_never_instrument));

When linking a binary, you can either manually link in the `XRay Runtime
Library`_ or use ``clang`` to link it in automatically with the
``-fxray-instrument`` flag.

LLVM Function Attribute
-----------------------

If you're using LLVM IR directly, you can add the ``function-instrument``
string attribute to your functions, to get the similar effect that the
C/C++/Objective-C source-level attributes would get:

.. code-block:: llvm

    define i32 @always_instrument() uwtable "function-instrument"="xray-always" {
      // ...
    }

    define i32 @never_instrument() uwtable "function-instrument"="xray-never" {
      // ...
    }

You can also set the ``xray-instruction-threshold`` attribute and provide a
numeric string value for how many instructions should be in the function before
it gets instrumented.

.. code-block:: llvm

    define i32 @maybe_instrument() uwtable "xray-instruction-threshold"="2" {
      // ...
    }

XRay Runtime Library
--------------------

The XRay Runtime Library is part of the compiler-rt project, which implements
the runtime components that perform the patching and unpatching of inserted
instrumentation points. When you use ``clang`` to link your binaries and the
``-fxray-instrument`` flag, it will automatically link in the XRay runtime.

The default implementation of the XRay runtime will enable XRay instrumentation
before ``main`` starts, which works for applications that have a short
lifetime. This implementation also records all function entry and exit events
which may result in a lot of records in the resulting trace.

Also by default the filename of the XRay trace is ``xray-log.XXXXXX`` where the
``XXXXXX`` part is randomly generated.

These options can be controlled through the ``XRAY_OPTIONS`` environment
variable, where we list down the options and their defaults below.

+-------------------+-----------------+---------------+------------------------+
| Option            | Type            | Default       | Description            |
+===================+=================+===============+========================+
| patch_premain     | ``bool``        | ``true``      | Whether to patch       |
|                   |                 |               | instrumentation points |
|                   |                 |               | before main.           |
+-------------------+-----------------+---------------+------------------------+
| xray_naive_log    | ``bool``        | ``true``      | Whether to install     |
|                   |                 |               | the naive log          |
|                   |                 |               | implementation.        |
+-------------------+-----------------+---------------+------------------------+
| xray_logfile_base | ``const char*`` | ``xray-log.`` | Filename base for the  |
|                   |                 |               | XRay logfile.          |
+-------------------+-----------------+---------------+------------------------+

If you choose to not use the default logging implementation that comes with the
XRay runtime and/or control when/how the XRay instrumentation runs, you may use
the XRay APIs directly for doing so. To do this, you'll need to include the
``xray_interface.h`` from the compiler-rt ``xray`` directory. The important API
functions we list below:

- ``__xray_set_handler(void (*entry)(int32_t, XRayEntryType))``: Install your
  own logging handler for when an event is encountered. See
  ``xray/xray_interface.h`` for more details.
- ``__xray_remove_handler()``: Removes whatever the installed handler is.
- ``__xray_patch()``: Patch all the instrumentation points defined in the
  binary.
- ``__xray_unpatch()``: Unpatch the instrumentation points defined in the
  binary.


Trace Analysis Tools
--------------------

We currently have the beginnings of a trace analysis tool in LLVM, which can be
found in the ``tools/llvm-xray`` directory. The ``llvm-xray`` tool currently
supports the following subcommands:

- ``extract``: Extract the instrumentation map from a binary, and return it as
  YAML.


Future Work
===========

There are a number of ongoing efforts for expanding the toolset building around
the XRay instrumentation system.

Flight Data Recorder Mode
-------------------------

The `XRay whitepaper`_ mentions a mode for when events are kept in memory, and
have the traces be dumped on demand through a triggering API. This work is
currently ongoing.

Trace Analysis
--------------

There are a few more subcommands making its way to the ``llvm-xray`` tool, that
are currently under review:

- ``convert``: Turns an XRay trace from one format to another. Currently
  supporting conversion from the binary XRay log to YAML.
- ``account``: Do function call accounting based on data in the XRay log.

We have more subcommands and modes that we're thinking of developing, in the
following forms:

- ``stack``: Reconstruct the function call stacks in a timeline.
- ``convert``: Converting from one version of the XRay log to another (higher)
  version, and converting to other trace formats (i.e. Chrome Trace Viewer,
  pprof, etc.).
- ``graph``: Generate a function call graph with relative timings and distributions.

More Platforms
--------------

Since XRay is only currently available in x86_64 and arm7 32-bit (no-thumb)
running Linux, we're looking to supporting more platforms (architectures and
operating systems).

.. References...

.. _`XRay whitepaper`: http://research.google.com/pubs/pub45287.html

