=====================
LLVM Coding Standards
=====================

.. contents::
   :local:

Introduction
============

This document attempts to describe a few coding standards that are being used in
the LLVM source tree.  Although no coding standards should be regarded as
absolute requirements to be followed in all instances, coding standards are
particularly important for large-scale code bases that follow a library-based
design (like LLVM).

While this document may provide guidance for some mechanical formatting issues,
whitespace, or other "microscopic details", these are not fixed standards.
Always follow the golden rule:

.. _Golden Rule:

    **If you are extending, enhancing, or bug fixing already implemented code,
    use the style that is already being used so that the source is uniform and
    easy to follow.**

Note that some code bases (e.g. ``libc++``) have really good reasons to deviate
from the coding standards.  In the case of ``libc++``, this is because the
naming and other conventions are dictated by the C++ standard.  If you think
there is a specific good reason to deviate from the standards here, please bring
it up on the LLVM-dev mailing list.

There are some conventions that are not uniformly followed in the code base
(e.g. the naming convention).  This is because they are relatively new, and a
lot of code was written before they were put in place.  Our long term goal is
for the entire codebase to follow the convention, but we explicitly *do not*
want patches that do large-scale reformatting of existing code.  On the other
hand, it is reasonable to rename the methods of a class if you're about to
change it in some other way.  Just do the reformatting as a separate commit
from the functionality change.
  
The ultimate goal of these guidelines is to increase the readability and
maintainability of our common source base. If you have suggestions for topics to
be included, please mail them to `Chris <mailto:sabre@nondot.org>`_.

Languages, Libraries, and Standards
===================================

Most source code in LLVM and other LLVM projects using these coding standards
is C++ code. There are some places where C code is used either due to
environment restrictions, historical restrictions, or due to third-party source
code imported into the tree. Generally, our preference is for standards
conforming, modern, and portable C++ code as the implementation language of
choice.

C++ Standard Versions
---------------------

LLVM, Clang, and LLD are currently written using C++11 conforming code,
although we restrict ourselves to features which are available in the major
toolchains supported as host compilers. The LLDB project is even more
aggressive in the set of host compilers supported and thus uses still more
features. Regardless of the supported features, code is expected to (when
reasonable) be standard, portable, and modern C++11 code. We avoid unnecessary
vendor-specific extensions, etc.

C++ Standard Library
--------------------

Use the C++ standard library facilities whenever they are available for
a particular task. LLVM and related projects emphasize and rely on the standard
library facilities for as much as possible. Common support libraries providing
functionality missing from the standard library for which there are standard
interfaces or active work on adding standard interfaces will often be
implemented in the LLVM namespace following the expected standard interface.

There are some exceptions such as the standard I/O streams library which are
avoided. Also, there is much more detailed information on these subjects in the
:doc:`ProgrammersManual`.

Supported C++11 Language and Library Features
---------------------------------------------

While LLVM, Clang, and LLD use C++11, not all features are available in all of
the toolchains which we support. The set of features supported for use in LLVM
is the intersection of those supported in the minimum requirements described
in the :doc:`GettingStarted` page, section `Software`.
The ultimate definition of this set is what build bots with those respective
toolchains accept. Don't argue with the build bots. However, we have some
guidance below to help you know what to expect.

Each toolchain provides a good reference for what it accepts:

* Clang: http://clang.llvm.org/cxx_status.html
* GCC: http://gcc.gnu.org/projects/cxx0x.html
* MSVC: http://msdn.microsoft.com/en-us/library/hh567368.aspx

In most cases, the MSVC list will be the dominating factor. Here is a summary
of the features that are expected to work. Features not on this list are
unlikely to be supported by our host compilers.

* Rvalue references: N2118_

  * But *not* Rvalue references for ``*this`` or member qualifiers (N2439_)

* Static assert: N1720_
* ``auto`` type deduction: N1984_, N1737_
* Trailing return types: N2541_
* Lambdas: N2927_

  * But *not* lambdas with default arguments.

* ``decltype``: N2343_
* Nested closing right angle brackets: N1757_
* Extern templates: N1987_
* ``nullptr``: N2431_
* Strongly-typed and forward declarable enums: N2347_, N2764_
* Local and unnamed types as template arguments: N2657_
* Range-based for-loop: N2930_

  * But ``{}`` are required around inner ``do {} while()`` loops.  As a result,
    ``{}`` are required around function-like macros inside range-based for
    loops.

* ``override`` and ``final``: N2928_, N3206_, N3272_
* Atomic operations and the C++11 memory model: N2429_
* Variadic templates: N2242_
* Explicit conversion operators: N2437_
* Defaulted and deleted functions: N2346_
* Initializer lists: N2627_
* Delegating constructors: N1986_
* Default member initializers (non-static data member initializers): N2756_

  * Feel free to use these wherever they make sense and where the `=`
    syntax is allowed. Don't use braced initialization syntax.

.. _N2118: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html
.. _N2439: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm
.. _N1720: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html
.. _N1984: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf
.. _N1737: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf
.. _N2541: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm
.. _N2927: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2009/n2927.pdf
.. _N2343: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf
.. _N1757: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html
.. _N1987: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm
.. _N2431: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf
.. _N2347: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf
.. _N2764: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf
.. _N2657: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm
.. _N2930: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2009/n2930.html
.. _N2928: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2009/n2928.htm
.. _N3206: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm
.. _N3272: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm
.. _N2429: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm
.. _N2242: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf
.. _N2437: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf
.. _N2346: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm
.. _N2627: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm
.. _N1986: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf
.. _N2756: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2756.htm

The supported features in the C++11 standard libraries are less well tracked,
but also much greater. Most of the standard libraries implement most of C++11's
library. The most likely lowest common denominator is Linux support. For
libc++, the support is just poorly tested and undocumented but expected to be
largely complete. YMMV. For libstdc++, the support is documented in detail in
`the libstdc++ manual`_. There are some very minor missing facilities that are
unlikely to be common problems, and there are a few larger gaps that are worth
being aware of:

* Not all of the type traits are implemented
* No regular expression library.
* While most of the atomics library is well implemented, the fences are
  missing. Fortunately, they are rarely needed.
* The locale support is incomplete.

Other than these areas you should assume the standard library is available and
working as expected until some build bot tells you otherwise. If you're in an
uncertain area of one of the above points, but you cannot test on a Linux
system, your best approach is to minimize your use of these features, and watch
the Linux build bots to find out if your usage triggered a bug. For example, if
you hit a type trait which doesn't work we can then add support to LLVM's
traits header to emulate it.

.. _the libstdc++ manual:
  http://gcc.gnu.org/onlinedocs/gcc-4.8.0/libstdc++/manual/manual/status.html#status.iso.2011

Other Languages
---------------

Any code written in the Go programming language is not subject to the
formatting rules below. Instead, we adopt the formatting rules enforced by
the `gofmt`_ tool.

Go code should strive to be idiomatic. Two good sets of guidelines for what
this means are `Effective Go`_ and `Go Code Review Comments`_.

.. _gofmt:
  https://golang.org/cmd/gofmt/

.. _Effective Go:
  https://golang.org/doc/effective_go.html

.. _Go Code Review Comments:
  https://github.com/golang/go/wiki/CodeReviewComments

Mechanical Source Issues
========================

Source Code Formatting
----------------------

Commenting
^^^^^^^^^^

Comments are one critical part of readability and maintainability.  Everyone
knows they should comment their code, and so should you.  When writing comments,
write them as English prose, which means they should use proper capitalization,
punctuation, etc.  Aim to describe what the code is trying to do and why, not
*how* it does it at a micro level. Here are a few critical things to document:

.. _header file comment:

File Headers
""""""""""""

Every source file should have a header on it that describes the basic purpose of
the file.  If a file does not have a header, it should not be checked into the
tree.  The standard header looks like this:

.. code-block:: c++

  //===-- llvm/Instruction.h - Instruction class definition -------*- C++ -*-===//
  //
  //                     The LLVM Compiler Infrastructure
  //
  // This file is distributed under the University of Illinois Open Source
  // License. See LICENSE.TXT for details.
  //
  //===----------------------------------------------------------------------===//
  ///
  /// \file
  /// This file contains the declaration of the Instruction class, which is the
  /// base class for all of the VM instructions.
  ///
  //===----------------------------------------------------------------------===//

A few things to note about this particular format: The "``-*- C++ -*-``" string
on the first line is there to tell Emacs that the source file is a C++ file, not
a C file (Emacs assumes ``.h`` files are C files by default).

.. note::

    This tag is not necessary in ``.cpp`` files.  The name of the file is also
    on the first line, along with a very short description of the purpose of the
    file.  This is important when printing out code and flipping though lots of
    pages.

The next section in the file is a concise note that defines the license that the
file is released under.  This makes it perfectly clear what terms the source
code can be distributed under and should not be modified in any way.

The main body is a ``doxygen`` comment (identified by the ``///`` comment
marker instead of the usual ``//``) describing the purpose of the file.  The
first sentence (or a passage beginning with ``\brief``) is used as an abstract.
Any additional information should be separated by a blank line.  If an
algorithm is being implemented or something tricky is going on, a reference
to the paper where it is published should be included, as well as any notes or
*gotchas* in the code to watch out for.

Class overviews
"""""""""""""""

Classes are one fundamental part of a good object oriented design.  As such, a
class definition should have a comment block that explains what the class is
used for and how it works.  Every non-trivial class is expected to have a
``doxygen`` comment block.

Method information
""""""""""""""""""

Methods defined in a class (as well as any global functions) should also be
documented properly.  A quick note about what it does and a description of the
borderline behaviour is all that is necessary here (unless something
particularly tricky or insidious is going on).  The hope is that people can
figure out how to use your interfaces without reading the code itself.

Good things to talk about here are what happens when something unexpected
happens: does the method return null?  Abort?  Format your hard disk?

Comment Formatting
^^^^^^^^^^^^^^^^^^

In general, prefer C++ style comments (``//`` for normal comments, ``///`` for
``doxygen`` documentation comments).  They take less space, require
less typing, don't have nesting problems, etc.  There are a few cases when it is
useful to use C style (``/* */``) comments however:

#. When writing C code: Obviously if you are writing C code, use C style
   comments.

#. When writing a header file that may be ``#include``\d by a C source file.

#. When writing a source file that is used by a tool that only accepts C style
   comments.

Commenting out large blocks of code is discouraged, but if you really have to do
this (for documentation purposes or as a suggestion for debug printing), use
``#if 0`` and ``#endif``. These nest properly and are better behaved in general
than C style comments.

Doxygen Use in Documentation Comments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the ``\file`` command to turn the standard file header into a file-level
comment.

Include descriptive paragraphs for all public interfaces (public classes,
member and non-member functions).  Don't just restate the information that can
be inferred from the API name.  The first sentence (or a paragraph beginning
with ``\brief``) is used as an abstract. Try to use a single sentence as the
``\brief`` adds visual clutter.  Put detailed discussion into separate
paragraphs.

To refer to parameter names inside a paragraph, use the ``\p name`` command.
Don't use the ``\arg name`` command since it starts a new paragraph that
contains documentation for the parameter.

Wrap non-inline code examples in ``\code ... \endcode``.

To document a function parameter, start a new paragraph with the
``\param name`` command.  If the parameter is used as an out or an in/out
parameter, use the ``\param [out] name`` or ``\param [in,out] name`` command,
respectively.

To describe function return value, start a new paragraph with the ``\returns``
command.

A minimal documentation comment:

.. code-block:: c++

  /// Sets the xyzzy property to \p Baz.
  void setXyzzy(bool Baz);

A documentation comment that uses all Doxygen features in a preferred way:

.. code-block:: c++

  /// Does foo and bar.
  ///
  /// Does not do foo the usual way if \p Baz is true.
  ///
  /// Typical usage:
  /// \code
  ///   fooBar(false, "quux", Res);
  /// \endcode
  ///
  /// \param Quux kind of foo to do.
  /// \param [out] Result filled with bar sequence on foo success.
  ///
  /// \returns true on success.
  bool fooBar(bool Baz, StringRef Quux, std::vector<int> &Result);

Don't duplicate the documentation comment in the header file and in the
implementation file.  Put the documentation comments for public APIs into the
header file.  Documentation comments for private APIs can go to the
implementation file.  In any case, implementation files can include additional
comments (not necessarily in Doxygen markup) to explain implementation details
as needed.

Don't duplicate function or class name at the beginning of the comment.
For humans it is obvious which function or class is being documented;
automatic documentation processing tools are smart enough to bind the comment
to the correct declaration.

Wrong:

.. code-block:: c++

  // In Something.h:

  /// Something - An abstraction for some complicated thing.
  class Something {
  public:
    /// fooBar - Does foo and bar.
    void fooBar();
  };

  // In Something.cpp:

  /// fooBar - Does foo and bar.
  void Something::fooBar() { ... }

Correct:

.. code-block:: c++

  // In Something.h:

  /// An abstraction for some complicated thing.
  class Something {
  public:
    /// Does foo and bar.
    void fooBar();
  };

  // In Something.cpp:

  // Builds a B-tree in order to do foo.  See paper by...
  void Something::fooBar() { ... }

It is not required to use additional Doxygen features, but sometimes it might
be a good idea to do so.

Consider:

* adding comments to any narrow namespace containing a collection of
  related functions or types;

* using top-level groups to organize a collection of related functions at
  namespace scope where the grouping is smaller than the namespace;

* using member groups and additional comments attached to member
  groups to organize within a class.

For example:

.. code-block:: c++

  class Something {
    /// \name Functions that do Foo.
    /// @{
    void fooBar();
    void fooBaz();
    /// @}
    ...
  };

``#include`` Style
^^^^^^^^^^^^^^^^^^

Immediately after the `header file comment`_ (and include guards if working on a
header file), the `minimal list of #includes`_ required by the file should be
listed.  We prefer these ``#include``\s to be listed in this order:

.. _Main Module Header:
.. _Local/Private Headers:

#. Main Module Header
#. Local/Private Headers
#. LLVM project/subproject headers (``clang/...``, ``lldb/...``, ``llvm/...``, etc)
#. System ``#include``\s

and each category should be sorted lexicographically by the full path.

The `Main Module Header`_ file applies to ``.cpp`` files which implement an
interface defined by a ``.h`` file.  This ``#include`` should always be included
**first** regardless of where it lives on the file system.  By including a
header file first in the ``.cpp`` files that implement the interfaces, we ensure
that the header does not have any hidden dependencies which are not explicitly
``#include``\d in the header, but should be. It is also a form of documentation
in the ``.cpp`` file to indicate where the interfaces it implements are defined.

LLVM project and subproject headers should be grouped from most specific to least
specific, for the same reasons described above.  For example, LLDB depends on
both clang and LLVM, and clang depends on LLVM.  So an LLDB source file should
include ``lldb`` headers first, followed by ``clang`` headers, followed by
``llvm`` headers, to reduce the possibility (for example) of an LLDB header
accidentally picking up a missing include due to the previous inclusion of that
header in the main source file or some earlier header file.  clang should
similarly include its own headers before including llvm headers.  This rule
applies to all LLVM subprojects.

.. _fit into 80 columns:

Source Code Width
^^^^^^^^^^^^^^^^^

Write your code to fit within 80 columns of text.  This helps those of us who
like to print out code and look at your code in an ``xterm`` without resizing
it.

The longer answer is that there must be some limit to the width of the code in
order to reasonably allow developers to have multiple files side-by-side in
windows on a modest display.  If you are going to pick a width limit, it is
somewhat arbitrary but you might as well pick something standard.  Going with 90
columns (for example) instead of 80 columns wouldn't add any significant value
and would be detrimental to printing out code.  Also many other projects have
standardized on 80 columns, so some people have already configured their editors
for it (vs something else, like 90 columns).

This is one of many contentious issues in coding standards, but it is not up for
debate.

Use Spaces Instead of Tabs
^^^^^^^^^^^^^^^^^^^^^^^^^^

In all cases, prefer spaces to tabs in source files.  People have different
preferred indentation levels, and different styles of indentation that they
like; this is fine.  What isn't fine is that different editors/viewers expand
tabs out to different tab stops.  This can cause your code to look completely
unreadable, and it is not worth dealing with.

As always, follow the `Golden Rule`_ above: follow the style of
existing code if you are modifying and extending it.  If you like four spaces of
indentation, **DO NOT** do that in the middle of a chunk of code with two spaces
of indentation.  Also, do not reindent a whole source file: it makes for
incredible diffs that are absolutely worthless.

Indent Code Consistently
^^^^^^^^^^^^^^^^^^^^^^^^

Okay, in your first year of programming you were told that indentation is
important. If you didn't believe and internalize this then, now is the time.
Just do it. With the introduction of C++11, there are some new formatting
challenges that merit some suggestions to help have consistent, maintainable,
and tool-friendly formatting and indentation.

Format Lambdas Like Blocks Of Code
""""""""""""""""""""""""""""""""""

When formatting a multi-line lambda, format it like a block of code, that's
what it is. If there is only one multi-line lambda in a statement, and there
are no expressions lexically after it in the statement, drop the indent to the
standard two space indent for a block of code, as if it were an if-block opened
by the preceding part of the statement:

.. code-block:: c++

  std::sort(foo.begin(), foo.end(), [&](Foo a, Foo b) -> bool {
    if (a.blah < b.blah)
      return true;
    if (a.baz < b.baz)
      return true;
    return a.bam < b.bam;
  });

To take best advantage of this formatting, if you are designing an API which
accepts a continuation or single callable argument (be it a functor, or
a ``std::function``), it should be the last argument if at all possible.

If there are multiple multi-line lambdas in a statement, or there is anything
interesting after the lambda in the statement, indent the block two spaces from
the indent of the ``[]``:

.. code-block:: c++

  dyn_switch(V->stripPointerCasts(),
             [] (PHINode *PN) {
               // process phis...
             },
             [] (SelectInst *SI) {
               // process selects...
             },
             [] (LoadInst *LI) {
               // process loads...
             },
             [] (AllocaInst *AI) {
               // process allocas...
             });

Braced Initializer Lists
""""""""""""""""""""""""

With C++11, there are significantly more uses of braced lists to perform
initialization. These allow you to easily construct aggregate temporaries in
expressions among other niceness. They now have a natural way of ending up
nested within each other and within function calls in order to build up
aggregates (such as option structs) from local variables. To make matters
worse, we also have many more uses of braces in an expression context that are
*not* performing initialization.

The historically common formatting of braced initialization of aggregate
variables does not mix cleanly with deep nesting, general expression contexts,
function arguments, and lambdas. We suggest new code use a simple rule for
formatting braced initialization lists: act as-if the braces were parentheses
in a function call. The formatting rules exactly match those already well
understood for formatting nested function calls. Examples:

.. code-block:: c++

  foo({a, b, c}, {1, 2, 3});

  llvm::Constant *Mask[] = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(getLLVMContext()), 0),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(getLLVMContext()), 1),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(getLLVMContext()), 2)};

This formatting scheme also makes it particularly easy to get predictable,
consistent, and automatic formatting with tools like `Clang Format`_.

.. _Clang Format: http://clang.llvm.org/docs/ClangFormat.html

Language and Compiler Issues
----------------------------

Treat Compiler Warnings Like Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your code has compiler warnings in it, something is wrong --- you aren't
casting values correctly, you have "questionable" constructs in your code, or
you are doing something legitimately wrong.  Compiler warnings can cover up
legitimate errors in output and make dealing with a translation unit difficult.

It is not possible to prevent all warnings from all compilers, nor is it
desirable.  Instead, pick a standard compiler (like ``gcc``) that provides a
good thorough set of warnings, and stick to it.  At least in the case of
``gcc``, it is possible to work around any spurious errors by changing the
syntax of the code slightly.  For example, a warning that annoys me occurs when
I write code like this:

.. code-block:: c++

  if (V = getValue()) {
    ...
  }

``gcc`` will warn me that I probably want to use the ``==`` operator, and that I
probably mistyped it.  In most cases, I haven't, and I really don't want the
spurious errors.  To fix this particular problem, I rewrite the code like
this:

.. code-block:: c++

  if ((V = getValue())) {
    ...
  }

which shuts ``gcc`` up.  Any ``gcc`` warning that annoys you can be fixed by
massaging the code appropriately.

Write Portable Code
^^^^^^^^^^^^^^^^^^^

In almost all cases, it is possible and within reason to write completely
portable code.  If there are cases where it isn't possible to write portable
code, isolate it behind a well defined (and well documented) interface.

In practice, this means that you shouldn't assume much about the host compiler
(and Visual Studio tends to be the lowest common denominator).  If advanced
features are used, they should only be an implementation detail of a library
which has a simple exposed API, and preferably be buried in ``libSystem``.

Do not use RTTI or Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In an effort to reduce code and executable size, LLVM does not use RTTI
(e.g. ``dynamic_cast<>;``) or exceptions.  These two language features violate
the general C++ principle of *"you only pay for what you use"*, causing
executable bloat even if exceptions are never used in the code base, or if RTTI
is never used for a class.  Because of this, we turn them off globally in the
code.

That said, LLVM does make extensive use of a hand-rolled form of RTTI that use
templates like :ref:`isa\<>, cast\<>, and dyn_cast\<> <isa>`.
This form of RTTI is opt-in and can be
:doc:`added to any class <HowToSetUpLLVMStyleRTTI>`. It is also
substantially more efficient than ``dynamic_cast<>``.

.. _static constructor:

Do not use Static Constructors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Static constructors and destructors (e.g. global variables whose types have a
constructor or destructor) should not be added to the code base, and should be
removed wherever possible.  Besides `well known problems
<http://yosefk.com/c++fqa/ctors.html#fqa-10.12>`_ where the order of
initialization is undefined between globals in different source files, the
entire concept of static constructors is at odds with the common use case of
LLVM as a library linked into a larger application.
  
Consider the use of LLVM as a JIT linked into another application (perhaps for
`OpenGL, custom languages <http://llvm.org/Users.html>`_, `shaders in movies
<http://llvm.org/devmtg/2010-11/Gritz-OpenShadingLang.pdf>`_, etc). Due to the
design of static constructors, they must be executed at startup time of the
entire application, regardless of whether or how LLVM is used in that larger
application.  There are two problems with this:

* The time to run the static constructors impacts startup time of applications
  --- a critical time for GUI apps, among others.
  
* The static constructors cause the app to pull many extra pages of memory off
  the disk: both the code for the constructor in each ``.o`` file and the small
  amount of data that gets touched. In addition, touched/dirty pages put more
  pressure on the VM system on low-memory machines.

We would really like for there to be zero cost for linking in an additional LLVM
target or other library into an application, but static constructors violate
this goal.
  
That said, LLVM unfortunately does contain static constructors.  It would be a
`great project <http://llvm.org/PR11944>`_ for someone to purge all static
constructors from LLVM, and then enable the ``-Wglobal-constructors`` warning
flag (when building with Clang) to ensure we do not regress in the future.

Use of ``class`` and ``struct`` Keywords
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In C++, the ``class`` and ``struct`` keywords can be used almost
interchangeably. The only difference is when they are used to declare a class:
``class`` makes all members private by default while ``struct`` makes all
members public by default.

Unfortunately, not all compilers follow the rules and some will generate
different symbols based on whether ``class`` or ``struct`` was used to declare
the symbol (e.g., MSVC).  This can lead to problems at link time.

* All declarations and definitions of a given ``class`` or ``struct`` must use
  the same keyword.  For example:

.. code-block:: c++

  class Foo;

  // Breaks mangling in MSVC.
  struct Foo { int Data; };

* As a rule of thumb, ``struct`` should be kept to structures where *all*
  members are declared public.

.. code-block:: c++

  // Foo feels like a class... this is strange.
  struct Foo {
  private:
    int Data;
  public:
    Foo() : Data(0) { }
    int getData() const { return Data; }
    void setData(int D) { Data = D; }
  };

  // Bar isn't POD, but it does look like a struct.
  struct Bar {
    int Data;
    Bar() : Data(0) { }
  };

Do not use Braced Initializer Lists to Call a Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In C++11 there is a "generalized initialization syntax" which allows calling
constructors using braced initializer lists. Do not use these to call
constructors with any interesting logic or if you care that you're calling some
*particular* constructor. Those should look like function calls using
parentheses rather than like aggregate initialization. Similarly, if you need
to explicitly name the type and call its constructor to create a temporary,
don't use a braced initializer list. Instead, use a braced initializer list
(without any type for temporaries) when doing aggregate initialization or
something notionally equivalent. Examples:

.. code-block:: c++

  class Foo {
  public:
    // Construct a Foo by reading data from the disk in the whizbang format, ...
    Foo(std::string filename);

    // Construct a Foo by looking up the Nth element of some global data ...
    Foo(int N);

    // ...
  };

  // The Foo constructor call is very deliberate, no braces.
  std::fill(foo.begin(), foo.end(), Foo("name"));

  // The pair is just being constructed like an aggregate, use braces.
  bar_map.insert({my_key, my_value});

If you use a braced initializer list when initializing a variable, use an equals before the open curly brace:

.. code-block:: c++

  int data[] = {0, 1, 2, 3};

Use ``auto`` Type Deduction to Make Code More Readable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some are advocating a policy of "almost always ``auto``" in C++11, however LLVM
uses a more moderate stance. Use ``auto`` if and only if it makes the code more
readable or easier to maintain. Don't "almost always" use ``auto``, but do use
``auto`` with initializers like ``cast<Foo>(...)`` or other places where the
type is already obvious from the context. Another time when ``auto`` works well
for these purposes is when the type would have been abstracted away anyways,
often behind a container's typedef such as ``std::vector<T>::iterator``.

Beware unnecessary copies with ``auto``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The convenience of ``auto`` makes it easy to forget that its default behavior
is a copy.  Particularly in range-based ``for`` loops, careless copies are
expensive.

As a rule of thumb, use ``auto &`` unless you need to copy the result, and use
``auto *`` when copying pointers.

.. code-block:: c++

  // Typically there's no reason to copy.
  for (const auto &Val : Container) { observe(Val); }
  for (auto &Val : Container) { Val.change(); }

  // Remove the reference if you really want a new copy.
  for (auto Val : Container) { Val.change(); saveSomewhere(Val); }

  // Copy pointers, but make it clear that they're pointers.
  for (const auto *Ptr : Container) { observe(*Ptr); }
  for (auto *Ptr : Container) { Ptr->change(); }

Beware of non-determinism due to ordering of pointers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general, there is no relative ordering among pointers. As a result,
when unordered containers like sets and maps are used with pointer keys
the iteration order is undefined. Hence, iterating such containers may
result in non-deterministic code generation. While the generated code
might not necessarily be "wrong code", this non-determinism might result
in unexpected runtime crashes or simply hard to reproduce bugs on the
customer side making it harder to debug and fix.

As a rule of thumb, in case an ordered result is expected, remember to
sort an unordered container before iteration. Or use ordered containers
like vector/MapVector/SetVector if you want to iterate pointer keys.

Style Issues
============

The High-Level Issues
---------------------

A Public Header File **is** a Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C++ doesn't do too well in the modularity department.  There is no real
encapsulation or data hiding (unless you use expensive protocol classes), but it
is what we have to work with.  When you write a public header file (in the LLVM
source tree, they live in the top level "``include``" directory), you are
defining a module of functionality.

Ideally, modules should be completely independent of each other, and their
header files should only ``#include`` the absolute minimum number of headers
possible. A module is not just a class, a function, or a namespace: it's a
collection of these that defines an interface.  This interface may be several
functions, classes, or data structures, but the important issue is how they work
together.

In general, a module should be implemented by one or more ``.cpp`` files.  Each
of these ``.cpp`` files should include the header that defines their interface
first.  This ensures that all of the dependences of the module header have been
properly added to the module header itself, and are not implicit.  System
headers should be included after user headers for a translation unit.

.. _minimal list of #includes:

``#include`` as Little as Possible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``#include`` hurts compile time performance.  Don't do it unless you have to,
especially in header files.

But wait! Sometimes you need to have the definition of a class to use it, or to
inherit from it.  In these cases go ahead and ``#include`` that header file.  Be
aware however that there are many cases where you don't need to have the full
definition of a class.  If you are using a pointer or reference to a class, you
don't need the header file.  If you are simply returning a class instance from a
prototyped function or method, you don't need it.  In fact, for most cases, you
simply don't need the definition of a class. And not ``#include``\ing speeds up
compilation.

It is easy to try to go too overboard on this recommendation, however.  You
**must** include all of the header files that you are using --- you can include
them either directly or indirectly through another header file.  To make sure
that you don't accidentally forget to include a header file in your module
header, make sure to include your module header **first** in the implementation
file (as mentioned above).  This way there won't be any hidden dependencies that
you'll find out about later.

Keep "Internal" Headers Private
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many modules have a complex implementation that causes them to use more than one
implementation (``.cpp``) file.  It is often tempting to put the internal
communication interface (helper classes, extra functions, etc) in the public
module header file.  Don't do this!

If you really need to do something like this, put a private header file in the
same directory as the source files, and include it locally.  This ensures that
your private interface remains private and undisturbed by outsiders.

.. note::

    It's okay to put extra implementation methods in a public class itself. Just
    make them private (or protected) and all is well.

.. _early exits:

Use Early Exits and ``continue`` to Simplify Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When reading code, keep in mind how much state and how many previous decisions
have to be remembered by the reader to understand a block of code.  Aim to
reduce indentation where possible when it doesn't make it more difficult to
understand the code.  One great way to do this is by making use of early exits
and the ``continue`` keyword in long loops.  As an example of using an early
exit from a function, consider this "bad" code:

.. code-block:: c++

  Value *doSomething(Instruction *I) {
    if (!isa<TerminatorInst>(I) &&
        I->hasOneUse() && doOtherThing(I)) {
      ... some long code ....
    }

    return 0;
  }

This code has several problems if the body of the ``'if'`` is large.  When
you're looking at the top of the function, it isn't immediately clear that this
*only* does interesting things with non-terminator instructions, and only
applies to things with the other predicates.  Second, it is relatively difficult
to describe (in comments) why these predicates are important because the ``if``
statement makes it difficult to lay out the comments.  Third, when you're deep
within the body of the code, it is indented an extra level.  Finally, when
reading the top of the function, it isn't clear what the result is if the
predicate isn't true; you have to read to the end of the function to know that
it returns null.

It is much preferred to format the code like this:

.. code-block:: c++

  Value *doSomething(Instruction *I) {
    // Terminators never need 'something' done to them because ... 
    if (isa<TerminatorInst>(I))
      return 0;

    // We conservatively avoid transforming instructions with multiple uses
    // because goats like cheese.
    if (!I->hasOneUse())
      return 0;

    // This is really just here for example.
    if (!doOtherThing(I))
      return 0;
    
    ... some long code ....
  }

This fixes these problems.  A similar problem frequently happens in ``for``
loops.  A silly example is something like this:

.. code-block:: c++

  for (Instruction &I : BB) {
    if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
      Value *LHS = BO->getOperand(0);
      Value *RHS = BO->getOperand(1);
      if (LHS != RHS) {
        ...
      }
    }
  }

When you have very, very small loops, this sort of structure is fine. But if it
exceeds more than 10-15 lines, it becomes difficult for people to read and
understand at a glance. The problem with this sort of code is that it gets very
nested very quickly. Meaning that the reader of the code has to keep a lot of
context in their brain to remember what is going immediately on in the loop,
because they don't know if/when the ``if`` conditions will have ``else``\s etc.
It is strongly preferred to structure the loop like this:

.. code-block:: c++

  for (Instruction &I : BB) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO) continue;

    Value *LHS = BO->getOperand(0);
    Value *RHS = BO->getOperand(1);
    if (LHS == RHS) continue;

    ...
  }

This has all the benefits of using early exits for functions: it reduces nesting
of the loop, it makes it easier to describe why the conditions are true, and it
makes it obvious to the reader that there is no ``else`` coming up that they
have to push context into their brain for.  If a loop is large, this can be a
big understandability win.

Don't use ``else`` after a ``return``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For similar reasons above (reduction of indentation and easier reading), please
do not use ``'else'`` or ``'else if'`` after something that interrupts control
flow --- like ``return``, ``break``, ``continue``, ``goto``, etc. For
example, this is *bad*:

.. code-block:: c++

  case 'J': {
    if (Signed) {
      Type = Context.getsigjmp_bufType();
      if (Type.isNull()) {
        Error = ASTContext::GE_Missing_sigjmp_buf;
        return QualType();
      } else {
        break;
      }
    } else {
      Type = Context.getjmp_bufType();
      if (Type.isNull()) {
        Error = ASTContext::GE_Missing_jmp_buf;
        return QualType();
      } else {
        break;
      }
    }
  }

It is better to write it like this:

.. code-block:: c++

  case 'J':
    if (Signed) {
      Type = Context.getsigjmp_bufType();
      if (Type.isNull()) {
        Error = ASTContext::GE_Missing_sigjmp_buf;
        return QualType();
      }
    } else {
      Type = Context.getjmp_bufType();
      if (Type.isNull()) {
        Error = ASTContext::GE_Missing_jmp_buf;
        return QualType();
      }
    }
    break;

Or better yet (in this case) as:

.. code-block:: c++

  case 'J':
    if (Signed)
      Type = Context.getsigjmp_bufType();
    else
      Type = Context.getjmp_bufType();
    
    if (Type.isNull()) {
      Error = Signed ? ASTContext::GE_Missing_sigjmp_buf :
                       ASTContext::GE_Missing_jmp_buf;
      return QualType();
    }
    break;

The idea is to reduce indentation and the amount of code you have to keep track
of when reading the code.
              
Turn Predicate Loops into Predicate Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is very common to write small loops that just compute a boolean value.  There
are a number of ways that people commonly write these, but an example of this
sort of thing is:

.. code-block:: c++

  bool FoundFoo = false;
  for (unsigned I = 0, E = BarList.size(); I != E; ++I)
    if (BarList[I]->isFoo()) {
      FoundFoo = true;
      break;
    }

  if (FoundFoo) {
    ...
  }

This sort of code is awkward to write, and is almost always a bad sign.  Instead
of this sort of loop, we strongly prefer to use a predicate function (which may
be `static`_) that uses `early exits`_ to compute the predicate.  We prefer the
code to be structured like this:

.. code-block:: c++

  /// \returns true if the specified list has an element that is a foo.
  static bool containsFoo(const std::vector<Bar*> &List) {
    for (unsigned I = 0, E = List.size(); I != E; ++I)
      if (List[I]->isFoo())
        return true;
    return false;
  }
  ...

  if (containsFoo(BarList)) {
    ...
  }

There are many reasons for doing this: it reduces indentation and factors out
code which can often be shared by other code that checks for the same predicate.
More importantly, it *forces you to pick a name* for the function, and forces
you to write a comment for it.  In this silly example, this doesn't add much
value.  However, if the condition is complex, this can make it a lot easier for
the reader to understand the code that queries for this predicate.  Instead of
being faced with the in-line details of how we check to see if the BarList
contains a foo, we can trust the function name and continue reading with better
locality.

The Low-Level Issues
--------------------

Name Types, Functions, Variables, and Enumerators Properly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Poorly-chosen names can mislead the reader and cause bugs. We cannot stress
enough how important it is to use *descriptive* names.  Pick names that match
the semantics and role of the underlying entities, within reason.  Avoid
abbreviations unless they are well known.  After picking a good name, make sure
to use consistent capitalization for the name, as inconsistency requires clients
to either memorize the APIs or to look it up to find the exact spelling.

In general, names should be in camel case (e.g. ``TextFileReader`` and
``isLValue()``).  Different kinds of declarations have different rules:

* **Type names** (including classes, structs, enums, typedefs, etc) should be
  nouns and start with an upper-case letter (e.g. ``TextFileReader``).

* **Variable names** should be nouns (as they represent state).  The name should
  be camel case, and start with an upper case letter (e.g. ``Leader`` or
  ``Boats``).
  
* **Function names** should be verb phrases (as they represent actions), and
  command-like function should be imperative.  The name should be camel case,
  and start with a lower case letter (e.g. ``openFile()`` or ``isFoo()``).

* **Enum declarations** (e.g. ``enum Foo {...}``) are types, so they should
  follow the naming conventions for types.  A common use for enums is as a
  discriminator for a union, or an indicator of a subclass.  When an enum is
  used for something like this, it should have a ``Kind`` suffix
  (e.g. ``ValueKind``).
  
* **Enumerators** (e.g. ``enum { Foo, Bar }``) and **public member variables**
  should start with an upper-case letter, just like types.  Unless the
  enumerators are defined in their own small namespace or inside a class,
  enumerators should have a prefix corresponding to the enum declaration name.
  For example, ``enum ValueKind { ... };`` may contain enumerators like
  ``VK_Argument``, ``VK_BasicBlock``, etc.  Enumerators that are just
  convenience constants are exempt from the requirement for a prefix.  For
  instance:

  .. code-block:: c++

      enum {
        MaxSize = 42,
        Density = 12
      };
  
As an exception, classes that mimic STL classes can have member names in STL's
style of lower-case words separated by underscores (e.g. ``begin()``,
``push_back()``, and ``empty()``). Classes that provide multiple
iterators should add a singular prefix to ``begin()`` and ``end()``
(e.g. ``global_begin()`` and ``use_begin()``).

Here are some examples of good and bad names:

.. code-block:: c++

  class VehicleMaker {
    ...
    Factory<Tire> F;            // Bad -- abbreviation and non-descriptive.
    Factory<Tire> Factory;      // Better.
    Factory<Tire> TireFactory;  // Even better -- if VehicleMaker has more than one
                                // kind of factories.
  };

  Vehicle makeVehicle(VehicleType Type) {
    VehicleMaker M;                         // Might be OK if having a short life-span.
    Tire Tmp1 = M.makeTire();               // Bad -- 'Tmp1' provides no information.
    Light Headlight = M.makeLight("head");  // Good -- descriptive.
    ...
  }

Assert Liberally
^^^^^^^^^^^^^^^^

Use the "``assert``" macro to its fullest.  Check all of your preconditions and
assumptions, you never know when a bug (not necessarily even yours) might be
caught early by an assertion, which reduces debugging time dramatically.  The
"``<cassert>``" header file is probably already included by the header files you
are using, so it doesn't cost anything to use it.

To further assist with debugging, make sure to put some kind of error message in
the assertion statement, which is printed if the assertion is tripped. This
helps the poor debugger make sense of why an assertion is being made and
enforced, and hopefully what to do about it.  Here is one complete example:

.. code-block:: c++

  inline Value *getOperand(unsigned I) {
    assert(I < Operands.size() && "getOperand() out of range!");
    return Operands[I];
  }

Here are more examples:

.. code-block:: c++

  assert(Ty->isPointerType() && "Can't allocate a non-pointer type!");

  assert((Opcode == Shl || Opcode == Shr) && "ShiftInst Opcode invalid!");

  assert(idx < getNumSuccessors() && "Successor # out of range!");

  assert(V1.getType() == V2.getType() && "Constant types must be identical!");

  assert(isa<PHINode>(Succ->front()) && "Only works on PHId BBs!");

You get the idea.

In the past, asserts were used to indicate a piece of code that should not be
reached.  These were typically of the form:

.. code-block:: c++

  assert(0 && "Invalid radix for integer literal");

This has a few issues, the main one being that some compilers might not
understand the assertion, or warn about a missing return in builds where
assertions are compiled out.

Today, we have something much better: ``llvm_unreachable``:

.. code-block:: c++

  llvm_unreachable("Invalid radix for integer literal");

When assertions are enabled, this will print the message if it's ever reached
and then exit the program. When assertions are disabled (i.e. in release
builds), ``llvm_unreachable`` becomes a hint to compilers to skip generating
code for this branch. If the compiler does not support this, it will fall back
to the "abort" implementation.

Neither assertions or ``llvm_unreachable`` will abort the program on a release
build. If the error condition can be triggered by user input then the
recoverable error mechanism described in :doc:`ProgrammersManual` should be
used instead. In cases where this is not practical, ``report_fatal_error`` may
be used.

Another issue is that values used only by assertions will produce an "unused
value" warning when assertions are disabled.  For example, this code will warn:

.. code-block:: c++

  unsigned Size = V.size();
  assert(Size > 42 && "Vector smaller than it should be");

  bool NewToSet = Myset.insert(Value);
  assert(NewToSet && "The value shouldn't be in the set yet");

These are two interesting different cases. In the first case, the call to
``V.size()`` is only useful for the assert, and we don't want it executed when
assertions are disabled.  Code like this should move the call into the assert
itself.  In the second case, the side effects of the call must happen whether
the assert is enabled or not.  In this case, the value should be cast to void to
disable the warning.  To be specific, it is preferred to write the code like
this:

.. code-block:: c++

  assert(V.size() > 42 && "Vector smaller than it should be");

  bool NewToSet = Myset.insert(Value); (void)NewToSet;
  assert(NewToSet && "The value shouldn't be in the set yet");

Do Not Use ``using namespace std``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In LLVM, we prefer to explicitly prefix all identifiers from the standard
namespace with an "``std::``" prefix, rather than rely on "``using namespace
std;``".

In header files, adding a ``'using namespace XXX'`` directive pollutes the
namespace of any source file that ``#include``\s the header.  This is clearly a
bad thing.

In implementation files (e.g. ``.cpp`` files), the rule is more of a stylistic
rule, but is still important.  Basically, using explicit namespace prefixes
makes the code **clearer**, because it is immediately obvious what facilities
are being used and where they are coming from. And **more portable**, because
namespace clashes cannot occur between LLVM code and other namespaces.  The
portability rule is important because different standard library implementations
expose different symbols (potentially ones they shouldn't), and future revisions
to the C++ standard will add more symbols to the ``std`` namespace.  As such, we
never use ``'using namespace std;'`` in LLVM.

The exception to the general rule (i.e. it's not an exception for the ``std``
namespace) is for implementation files.  For example, all of the code in the
LLVM project implements code that lives in the 'llvm' namespace.  As such, it is
ok, and actually clearer, for the ``.cpp`` files to have a ``'using namespace
llvm;'`` directive at the top, after the ``#include``\s.  This reduces
indentation in the body of the file for source editors that indent based on
braces, and keeps the conceptual context cleaner.  The general form of this rule
is that any ``.cpp`` file that implements code in any namespace may use that
namespace (and its parents'), but should not use any others.

Provide a Virtual Method Anchor for Classes in Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a class is defined in a header file and has a vtable (either it has virtual
methods or it derives from classes with virtual methods), it must always have at
least one out-of-line virtual method in the class.  Without this, the compiler
will copy the vtable and RTTI into every ``.o`` file that ``#include``\s the
header, bloating ``.o`` file sizes and increasing link times.

Don't use default labels in fully covered switches over enumerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``-Wswitch`` warns if a switch, without a default label, over an enumeration
does not cover every enumeration value. If you write a default label on a fully
covered switch over an enumeration then the ``-Wswitch`` warning won't fire
when new elements are added to that enumeration. To help avoid adding these
kinds of defaults, Clang has the warning ``-Wcovered-switch-default`` which is
off by default but turned on when building LLVM with a version of Clang that
supports the warning.

A knock-on effect of this stylistic requirement is that when building LLVM with
GCC you may get warnings related to "control may reach end of non-void function"
if you return from each case of a covered switch-over-enum because GCC assumes
that the enum expression may take any representable value, not just those of
individual enumerators. To suppress this warning, use ``llvm_unreachable`` after
the switch.

Use range-based ``for`` loops wherever possible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The introduction of range-based ``for`` loops in C++11 means that explicit
manipulation of iterators is rarely necessary. We use range-based ``for``
loops wherever possible for all newly added code. For example:

.. code-block:: c++

  BasicBlock *BB = ...
  for (Instruction &I : *BB)
    ... use I ...

Don't evaluate ``end()`` every time through a loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In cases where range-based ``for`` loops can't be used and it is necessary
to write an explicit iterator-based loop, pay close attention to whether
``end()`` is re-evaluted on each loop iteration. One common mistake is to
write a loop in this style:

.. code-block:: c++

  BasicBlock *BB = ...
  for (auto I = BB->begin(); I != BB->end(); ++I)
    ... use I ...

The problem with this construct is that it evaluates "``BB->end()``" every time
through the loop.  Instead of writing the loop like this, we strongly prefer
loops to be written so that they evaluate it once before the loop starts.  A
convenient way to do this is like so:

.. code-block:: c++

  BasicBlock *BB = ...
  for (auto I = BB->begin(), E = BB->end(); I != E; ++I)
    ... use I ...

The observant may quickly point out that these two loops may have different
semantics: if the container (a basic block in this case) is being mutated, then
"``BB->end()``" may change its value every time through the loop and the second
loop may not in fact be correct.  If you actually do depend on this behavior,
please write the loop in the first form and add a comment indicating that you
did it intentionally.

Why do we prefer the second form (when correct)?  Writing the loop in the first
form has two problems. First it may be less efficient than evaluating it at the
start of the loop.  In this case, the cost is probably minor --- a few extra
loads every time through the loop.  However, if the base expression is more
complex, then the cost can rise quickly.  I've seen loops where the end
expression was actually something like: "``SomeMap[X]->end()``" and map lookups
really aren't cheap.  By writing it in the second form consistently, you
eliminate the issue entirely and don't even have to think about it.

The second (even bigger) issue is that writing the loop in the first form hints
to the reader that the loop is mutating the container (a fact that a comment
would handily confirm!).  If you write the loop in the second form, it is
immediately obvious without even looking at the body of the loop that the
container isn't being modified, which makes it easier to read the code and
understand what it does.

While the second form of the loop is a few extra keystrokes, we do strongly
prefer it.

``#include <iostream>`` is Forbidden
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The use of ``#include <iostream>`` in library files is hereby **forbidden**,
because many common implementations transparently inject a `static constructor`_
into every translation unit that includes it.
  
Note that using the other stream headers (``<sstream>`` for example) is not
problematic in this regard --- just ``<iostream>``. However, ``raw_ostream``
provides various APIs that are better performing for almost every use than
``std::ostream`` style APIs.

.. note::

  New code should always use `raw_ostream`_ for writing, or the
  ``llvm::MemoryBuffer`` API for reading files.

.. _raw_ostream:

Use ``raw_ostream``
^^^^^^^^^^^^^^^^^^^

LLVM includes a lightweight, simple, and efficient stream implementation in
``llvm/Support/raw_ostream.h``, which provides all of the common features of
``std::ostream``.  All new code should use ``raw_ostream`` instead of
``ostream``.

Unlike ``std::ostream``, ``raw_ostream`` is not a template and can be forward
declared as ``class raw_ostream``.  Public headers should generally not include
the ``raw_ostream`` header, but use forward declarations and constant references
to ``raw_ostream`` instances.

Avoid ``std::endl``
^^^^^^^^^^^^^^^^^^^

The ``std::endl`` modifier, when used with ``iostreams`` outputs a newline to
the output stream specified.  In addition to doing this, however, it also
flushes the output stream.  In other words, these are equivalent:

.. code-block:: c++

  std::cout << std::endl;
  std::cout << '\n' << std::flush;

Most of the time, you probably have no reason to flush the output stream, so
it's better to use a literal ``'\n'``.

Don't use ``inline`` when defining a function in a class definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A member function defined in a class definition is implicitly inline, so don't
put the ``inline`` keyword in this case.

Don't:

.. code-block:: c++

  class Foo {
  public:
    inline void bar() {
      // ...
    }
  };

Do:

.. code-block:: c++

  class Foo {
  public:
    void bar() {
      // ...
    }
  };

Microscopic Details
-------------------

This section describes preferred low-level formatting guidelines along with
reasoning on why we prefer them.

Spaces Before Parentheses
^^^^^^^^^^^^^^^^^^^^^^^^^

We prefer to put a space before an open parenthesis only in control flow
statements, but not in normal function call expressions and function-like
macros.  For example, this is good:

.. code-block:: c++

  if (X) ...
  for (I = 0; I != 100; ++I) ...
  while (LLVMRocks) ...

  somefunc(42);
  assert(3 != 4 && "laws of math are failing me");
  
  A = foo(42, 92) + bar(X);

and this is bad:

.. code-block:: c++

  if(X) ...
  for(I = 0; I != 100; ++I) ...
  while(LLVMRocks) ...

  somefunc (42);
  assert (3 != 4 && "laws of math are failing me");
  
  A = foo (42, 92) + bar (X);

The reason for doing this is not completely arbitrary.  This style makes control
flow operators stand out more, and makes expressions flow better. The function
call operator binds very tightly as a postfix operator.  Putting a space after a
function name (as in the last example) makes it appear that the code might bind
the arguments of the left-hand-side of a binary operator with the argument list
of a function and the name of the right side.  More specifically, it is easy to
misread the "``A``" example as:

.. code-block:: c++

  A = foo ((42, 92) + bar) (X);

when skimming through the code.  By avoiding a space in a function, we avoid
this misinterpretation.

Prefer Preincrement
^^^^^^^^^^^^^^^^^^^

Hard fast rule: Preincrement (``++X``) may be no slower than postincrement
(``X++``) and could very well be a lot faster than it.  Use preincrementation
whenever possible.

The semantics of postincrement include making a copy of the value being
incremented, returning it, and then preincrementing the "work value".  For
primitive types, this isn't a big deal. But for iterators, it can be a huge
issue (for example, some iterators contains stack and set objects in them...
copying an iterator could invoke the copy ctor's of these as well).  In general,
get in the habit of always using preincrement, and you won't have a problem.


Namespace Indentation
^^^^^^^^^^^^^^^^^^^^^

In general, we strive to reduce indentation wherever possible.  This is useful
because we want code to `fit into 80 columns`_ without wrapping horribly, but
also because it makes it easier to understand the code. To facilitate this and
avoid some insanely deep nesting on occasion, don't indent namespaces. If it
helps readability, feel free to add a comment indicating what namespace is
being closed by a ``}``.  For example:

.. code-block:: c++

  namespace llvm {
  namespace knowledge {

  /// This class represents things that Smith can have an intimate
  /// understanding of and contains the data associated with it.
  class Grokable {
  ...
  public:
    explicit Grokable() { ... }
    virtual ~Grokable() = 0;
  
    ...

  };

  } // end namespace knowledge
  } // end namespace llvm


Feel free to skip the closing comment when the namespace being closed is
obvious for any reason. For example, the outer-most namespace in a header file
is rarely a source of confusion. But namespaces both anonymous and named in
source files that are being closed half way through the file probably could use
clarification.

.. _static:

Anonymous Namespaces
^^^^^^^^^^^^^^^^^^^^

After talking about namespaces in general, you may be wondering about anonymous
namespaces in particular.  Anonymous namespaces are a great language feature
that tells the C++ compiler that the contents of the namespace are only visible
within the current translation unit, allowing more aggressive optimization and
eliminating the possibility of symbol name collisions.  Anonymous namespaces are
to C++ as "static" is to C functions and global variables.  While "``static``"
is available in C++, anonymous namespaces are more general: they can make entire
classes private to a file.

The problem with anonymous namespaces is that they naturally want to encourage
indentation of their body, and they reduce locality of reference: if you see a
random function definition in a C++ file, it is easy to see if it is marked
static, but seeing if it is in an anonymous namespace requires scanning a big
chunk of the file.

Because of this, we have a simple guideline: make anonymous namespaces as small
as possible, and only use them for class declarations.  For example, this is
good:

.. code-block:: c++

  namespace {
  class StringSort {
  ...
  public:
    StringSort(...)
    bool operator<(const char *RHS) const;
  };
  } // end anonymous namespace

  static void runHelper() { 
    ... 
  }

  bool StringSort::operator<(const char *RHS) const {
    ...
  }

This is bad:

.. code-block:: c++

  namespace {

  class StringSort {
  ...
  public:
    StringSort(...)
    bool operator<(const char *RHS) const;
  };

  void runHelper() { 
    ... 
  }

  bool StringSort::operator<(const char *RHS) const {
    ...
  }

  } // end anonymous namespace

This is bad specifically because if you're looking at "``runHelper``" in the middle
of a large C++ file, that you have no immediate way to tell if it is local to
the file.  When it is marked static explicitly, this is immediately obvious.
Also, there is no reason to enclose the definition of "``operator<``" in the
namespace just because it was declared there.

See Also
========

A lot of these comments and recommendations have been culled from other sources.
Two particularly important books for our work are:

#. `Effective C++
   <http://www.amazon.com/Effective-Specific-Addison-Wesley-Professional-Computing/dp/0321334876>`_
   by Scott Meyers.  Also interesting and useful are "More Effective C++" and
   "Effective STL" by the same author.

#. `Large-Scale C++ Software Design
   <http://www.amazon.com/Large-Scale-Software-Design-John-Lakos/dp/0201633620/ref=sr_1_1>`_
   by John Lakos

If you get some free time, and you haven't read them: do so, you might learn
something.
