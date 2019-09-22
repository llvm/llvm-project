About
========

.. warning::

   If you are using a released version of LLVM, see `the download page
   <http://llvm.org/releases/>`_ to find your documentation.

The LLVM compiler infrastructure supports a wide range of projects, from
industrial strength compilers to specialized JIT applications to small
research projects.

Similarly, documentation is broken down into several high-level groupings
targeted at different audiences:

LLVM Design & Overview
======================

Several introductory papers and presentations.

.. toctree::
   :hidden:

   FAQ
   Lexicon

`Introduction to the LLVM Compiler`__
  Presentation providing a users introduction to LLVM.

  .. __: http://llvm.org/pubs/2008-10-04-ACAT-LLVM-Intro.html

`Intro to LLVM`__
  Book chapter providing a compiler hacker's introduction to LLVM.

  .. __: http://www.aosabook.org/en/llvm.html


`LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation`__
  Design overview.

  .. __: http://llvm.org/pubs/2004-01-30-CGO-LLVM.html

`LLVM: An Infrastructure for Multi-Stage Optimization`__
  More details (quite old now).

  .. __: http://llvm.org/pubs/2002-12-LattnerMSThesis.html

`Publications mentioning LLVM <http://llvm.org/pubs>`_
   ..

Documentation
=============

Getting Started, How-tos, Developer Guides, and Tutorials.

.. toctree::
   :hidden:

   UserGuides
   ProgrammingDocumentation
   SubsystemDocumentation

:doc:`UserGuides`
  For those new to the LLVM system.

:doc:`ProgrammingDocumentation`
  For developers of applications which use LLVM as a library.

:doc:`SubsystemDocumentation`
  For API clients and LLVM developers.

Getting Started/Tutorials
-------------------------

.. toctree::
   :hidden:

   GettingStarted
   tutorial/index
   GettingStartedVS

:doc:`GettingStarted`
   Discusses how to get up and running quickly with the LLVM infrastructure.
   Everything from unpacking and compilation of the distribution to execution
   of some tools.

:doc:`tutorial/index`
   Tutorials about using LLVM. Includes a tutorial about making a custom
   language with LLVM.

:doc:`GettingStartedVS`
   An addendum to the main Getting Started guide for those using Visual Studio
   on Windows.

Reference
---------

LLVM and API reference documentation.

.. toctree::
   :hidden:

   LangRef
   CommandGuide/index
   TestingGuide
   CompilerWriterInfo
   MIRLangRef

:doc:`LLVM Language Reference Manual <LangRef>`
  Defines the LLVM intermediate representation and the assembly form of the
  different nodes.

:doc:`LLVM Command Guide <CommandGuide/index>`
   A reference manual for the LLVM command line utilities ("man" pages for LLVM
   tools).

:doc:`LLVM Testing Infrastructure Guide <TestingGuide>`
   A reference manual for using the LLVM testing infrastructure.

:doc:`CompilerWriterInfo`
  A list of helpful links for compiler writers.

:doc:`Machine IR (MIR) Format Reference Manual <MIRLangRef>`
   A reference manual for the MIR serialization format, which is used to test
   LLVM's code generation passes.

`Doxygen generated documentation <http://llvm.org/doxygen/>`_
  (`classes <http://llvm.org/doxygen/inherits.html>`_)

`Documentation for Go bindings <http://godoc.org/llvm.org/llvm/bindings/go/llvm>`_

`Github Source Repository Browser <http://github.com/llvm/llvm-project//>`_
   ..

Community
=========

LLVM welcomes contributions of all kinds. To learn more, see the following articles:

.. toctree::
   :hidden:

   GettingInvolved

* :doc:`GettingInvolved`
* :ref:`development-process`
* :ref:`mailing-lists`
* :ref:`meetups-social-events`
* :ref:`community-proposals`

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
