===============
LLVMBuild Guide
===============

.. contents::
   :local:

Introduction
============

This document describes the ``LLVMBuild`` organization and files which
we use to describe parts of the LLVM ecosystem. For description of
specific LLVMBuild related tools, please see the command guide.

LLVM is designed to be a modular set of libraries which can be flexibly
mixed together in order to build a variety of tools, like compilers,
JITs, custom code generators, optimization passes, interpreters, and so
on. Related projects in the LLVM system like Clang and LLDB also tend to
follow this philosophy.

In order to support this usage style, LLVM has a fairly strict structure
as to how the source code and various components are organized. The
``LLVMBuild.txt`` files are the explicit specification of that
structure, and are used by the build systems and other tools in order to
develop the LLVM project.

Project Organization
====================

The source code for LLVM projects using the LLVMBuild system (LLVM,
Clang, and LLDB) is organized into *components*, which define the
separate pieces of functionality that make up the project. These
projects may consist of many libraries, associated tools, build tools,
or other utility tools (for example, testing tools).

For the most part, the project contents are organized around defining
one main component per each subdirectory. Each such directory contains
an ``LLVMBuild.txt`` which contains the component definitions.

The component descriptions for the project as a whole are automatically
gathered by the LLVMBuild tools. The tools automatically traverse the
source directory structure to find all of the component description
files. NOTE: For performance/sanity reasons, we only traverse into
subdirectories when the parent itself contains an ``LLVMBuild.txt``
description file.

Build Integration
=================

The LLVMBuild files themselves are just a declarative way to describe
the project structure. The actual building of the LLVM project is
handled by another build system (See: :doc:`CMake <CMake>`).

The build system implementation will load the relevant contents of the
LLVMBuild files and use that to drive the actual project build.
Typically, the build system will only need to load this information at
"configure" time, and use it to generate native information. Build
systems will also handle automatically reconfiguring their information
when the contents of the ``LLVMBuild.txt`` files change.

Developers generally are not expected to need to be aware of the details
of how the LLVMBuild system is integrated into their build. Ideally,
LLVM developers who are not working on the build system would only ever
need to modify the contents of the ``LLVMBuild.txt`` description files
(although we have not reached this goal yet).

For more information on the utility tool we provide to help interfacing
with the build system, please see the :doc:`llvm-build
<CommandGuide/llvm-build>` documentation.

Component Overview
==================

As mentioned earlier, LLVM projects are organized into logical
*components*. Every component is typically grouped into its own
subdirectory. Generally, a component is organized around a coherent
group of sources which have some kind of clear API separation from other
parts of the code.

LLVM primarily uses the following types of components:

- *Libraries* - Library components define a distinct API which can be
  independently linked into LLVM client applications. Libraries typically
  have private and public header files, and may specify a link of required
  libraries that they build on top of.
- *Build Tools* - Build tools are applications which are designed to be run
  as part of the build process (typically to generate other source files).
  Currently, LLVM uses one main build tool called :doc:`TableGen/index`
  to generate a variety of source files.
- *Tools* - Command line applications which are built using the LLVM
  component libraries. Most LLVM tools are small and are primarily
  frontends to the library interfaces.

Components are described using ``LLVMBuild.txt`` files in the directories
that define the component. See the `LLVMBuild Format Reference`_ section
for information on the exact format of these files.

LLVMBuild Format Reference
==========================

LLVMBuild files are written in a simple variant of the INI or configuration
file format (`Wikipedia entry`_). The format defines a list of sections
each of which may contain some number of properties. A simple example of
the file format is below:

.. _Wikipedia entry: http://en.wikipedia.org/wiki/INI_file

.. code-block:: ini

   ; Comments start with a semi-colon.

   ; Sections are declared using square brackets.
   [component_0]

   ; Properties are declared using '=' and are contained in the previous section.
   ;
   ; We support simple string and boolean scalar values and list values, where
   ; items are separated by spaces. There is no support for quoting, and so
   ; property values may not contain spaces.
   property_name = property_value
   list_property_name = value_1 value_2 ... value_n
   boolean_property_name = 1 (or 0)

LLVMBuild files are expected to define a strict set of sections and
properties. A typical component description file for a library
component would look like the following example:

.. code-block:: ini

   [component_0]
   type = Library
   name = Linker
   parent = Libraries
   required_libraries = Archive BitReader Core Support TransformUtils

A full description of the exact sections and properties which are
allowed follows.

Each file may define exactly one common component, named ``common``. The
common component may define the following properties:

-  ``subdirectories`` **[optional]**

   If given, a list of the names of the subdirectories from the current
   subpath to search for additional LLVMBuild files.

Each file may define multiple components. Each component is described by a
section who name starts with ``component``. The remainder of the section
name is ignored, but each section name must be unique. Typically components
are just number in order for files with multiple components
(``component_0``, ``component_1``, and so on).

.. warning::

   Section names not matching this format (or the ``common`` section) are
   currently unused and are disallowed.

Every component is defined by the properties in the section. The exact
list of properties that are allowed depends on the component type.
Components **may not** define any properties other than those expected
by the component type.

Every component must define the following properties:

-  ``type`` **[required]**

   The type of the component. Supported component types are detailed
   below. Most components will define additional properties which may be
   required or optional.

-  ``name`` **[required]**

   The name of the component. Names are required to be unique across the
   entire project.

-  ``parent`` **[required]**

   The name of the logical parent of the component. Components are
   organized into a logical tree to make it easier to navigate and
   organize groups of components. The parents have no semantics as far
   as the project build is concerned, however. Typically, the parent
   will be the main component of the parent directory.

   Components may reference the root pseudo component using ``$ROOT`` to
   indicate they should logically be grouped at the top-level.

Components may define the following properties:

-  ``dependencies`` **[optional]**

   If specified, a list of names of components which *must* be built
   prior to this one. This should only be exactly those components which
   produce some tool or source code required for building the component.

   .. note::

      ``Group`` and ``LibraryGroup`` components have no semantics for the
      actual build, and are not allowed to specify dependencies.

The following section lists the available component types, as well as
the properties which are associated with that component.

-  ``type = Group``

   Group components exist purely to allow additional arbitrary structuring
   of the logical components tree. For example, one might define a
   ``Libraries`` group to hold all of the root library components.

   ``Group`` components have no additionally properties.

-  ``type = Library``

   Library components define an individual library which should be built
   from the source code in the component directory.

   Components with this type use the following properties:

   -  ``library_name`` **[optional]**

      If given, the name to use for the actual library file on disk. If
      not given, the name is derived from the component name itself.

   -  ``required_libraries`` **[optional]**

      If given, a list of the names of ``Library`` or ``LibraryGroup``
      components which must also be linked in whenever this library is
      used. That is, the link time dependencies for this component. When
      tools are built, the build system will include the transitive closure
      of all ``required_libraries`` for the components the tool needs.

   -  ``add_to_library_groups`` **[optional]**

      If given, a list of the names of ``LibraryGroup`` components which
      this component is also part of. This allows nesting groups of
      components.  For example, the ``X86`` target might define a library
      group for all of the ``X86`` components. That library group might
      then be included in the ``all-targets`` library group.

   -  ``installed`` **[optional]** **[boolean]**

      Whether this library is installed. Libraries that are not installed
      are only reported by ``llvm-config`` when it is run as part of a
      development directory.

-  ``type = LibraryGroup``

   ``LibraryGroup`` components are a mechanism to allow easy definition of
   useful sets of related components. In particular, we use them to easily
   specify things like "all targets", or "all assembly printers".

   Components with this type use the following properties:

   -  ``required_libraries`` **[optional]**

      See the ``Library`` type for a description of this property.

   -  ``add_to_library_groups`` **[optional]**

      See the ``Library`` type for a description of this property.

-  ``type = TargetGroup``

   ``TargetGroup`` components are an extension of ``LibraryGroup``\s,
   specifically for defining LLVM targets (which are handled specially in a
   few places).

   The name of the component should always be the name of the target.

   Components with this type use the ``LibraryGroup`` properties in
   addition to:

   -  ``has_asmparser`` **[optional]** **[boolean]**

      Whether this target defines an assembly parser.

   -  ``has_asmprinter`` **[optional]** **[boolean]**

      Whether this target defines an assembly printer.

   -  ``has_disassembler`` **[optional]** **[boolean]**

      Whether this target defines a disassembler.

   -  ``has_jit`` **[optional]** **[boolean]**

      Whether this target supports JIT compilation.

-  ``type = Tool``

   ``Tool`` components define standalone command line tools which should be
   built from the source code in the component directory and linked.

   Components with this type use the following properties:

   -  ``required_libraries`` **[optional]**

      If given, a list of the names of ``Library`` or ``LibraryGroup``
      components which this tool is required to be linked with.

      .. note::

         The values should be the component names, which may not always
         match up with the actual library names on disk.

      Build systems are expected to properly include all of the libraries
      required by the linked components (i.e., the transitive closure of
      ``required_libraries``).

      Build systems are also expected to understand that those library
      components must be built prior to linking -- they do not also need
      to be listed under ``dependencies``.

-  ``type = BuildTool``

   ``BuildTool`` components are like ``Tool`` components, except that the
   tool is supposed to be built for the platform where the build is running
   (instead of that platform being targeted). Build systems are expected
   to handle the fact that required libraries may need to be built for
   multiple platforms in order to be able to link this tool.

   ``BuildTool`` components currently use the exact same properties as
   ``Tool`` components, the type distinction is only used to differentiate
   what the tool is built for.
