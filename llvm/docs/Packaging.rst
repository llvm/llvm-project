.. _packaging:

========================
Advice on Packaging LLVM
========================

.. contents::
   :local:

Overview
========

LLVM sets certain default configure options to make sure our developers don't
break things for constrained platforms.  These settings are not optimal for most
desktop systems, and we hope that packagers (e.g., Redhat, Debian, MacPorts,
etc.) will tweak them.  This document lists settings we suggest you tweak.

LLVM's API changes with each release, so users are likely to want, for example,
both LLVM-2.6 and LLVM-2.7 installed at the same time to support apps developed
against each.

Compile Flags
=============

LLVM runs much more quickly when it's optimized and assertions are removed.
However, such a build is currently incompatible with users who build without
defining ``NDEBUG``, and the lack of assertions makes it hard to debug problems
in user code.  We recommend allowing users to install both optimized and debug
versions of LLVM in parallel.  The following configure flags are relevant:

``--disable-assertions``
    Builds LLVM with ``NDEBUG`` defined.  Changes the LLVM ABI.  Also available
    by setting ``DISABLE_ASSERTIONS=0|1`` in ``make``'s environment.  This
    defaults to enabled regardless of the optimization setting, but it slows
    things down.

``--enable-debug-symbols``
    Builds LLVM with ``-g``.  Also available by setting ``DEBUG_SYMBOLS=0|1`` in
    ``make``'s environment.  This defaults to disabled when optimizing, so you
    should turn it back on to let users debug their programs.

``--enable-optimized``
    (For svn checkouts) Builds LLVM with ``-O2`` and, by default, turns off
    debug symbols.  Also available by setting ``ENABLE_OPTIMIZED=0|1`` in
    ``make``'s environment.  This defaults to enabled when not in a
    checkout.

C++ Features
============

RTTI
    LLVM disables RTTI by default.  Add ``REQUIRES_RTTI=1`` to your environment
    while running ``make`` to re-enable it.  This will allow users to build with
    RTTI enabled and still inherit from LLVM classes.

Shared Library
==============

Configure with ``--enable-shared`` to build
``libLLVM-<major>.<minor>.(so|dylib)`` and link the tools against it.  This
saves lots of binary size at the cost of some startup time.

Dependencies
============

``--enable-libffi``
    Depend on `libffi <http://sources.redhat.com/libffi/>`_ to allow the LLVM
    interpreter to call external functions.

``--with-oprofile``

    Depend on `libopagent
    <http://oprofile.sourceforge.net/doc/devel/index.html>`_ (>=version 0.9.4)
    to let the LLVM JIT tell oprofile about function addresses and line
    numbers.
