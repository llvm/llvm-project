.. role:: raw-html(raw)
   :format: html

=================================
LLVM Code Coverage Mapping Format
=================================

.. contents::
   :local:

Introduction
============

LLVM's code coverage mapping format is used to provide code coverage
analysis using LLVM's and Clang's instrumenation based profiling
(Clang's ``-fprofile-instr-generate`` option).

This document is aimed at those who use LLVM's code coverage mapping to provide
code coverage analysis for their own programs, and for those who would like
to know how it works under the hood. A prior knowledge of how Clang's profile
guided optimization works is useful, but not required.

We start by showing how to use LLVM and Clang for code coverage analysis,
then we briefly desribe LLVM's code coverage mapping format and the
way that Clang and LLVM's code coverage tool work with this format. After
the basics are down, more advanced features of the coverage mapping format
are discussed - such as the data structures, LLVM IR representation and
the binary encoding.

Quick Start
===========

Here's a short story that describes how to generate code coverage overview
for a sample source file called *test.c*.

* First, compile an instrumented version of your program using Clang's
  ``-fprofile-instr-generate`` option with the additional ``-fcoverage-mapping``
  option:

  ``clang -o test -fprofile-instr-generate -fcoverage-mapping test.c``
* Then, run the instrumented binary. The runtime will produce a file called
  *default.profraw* containing the raw profile instrumentation data:

  ``./test``
* After that, merge the profile data using the *llvm-profdata* tool:

  ``llvm-profdata merge -o test.profdata default.profraw``
* Finally, run LLVM's code coverage tool (*llvm-cov*) to produce the code
  coverage overview for the sample source file:

  ``llvm-cov show ./test -instr-profile=test.profdata test.c``

High Level Overview
===================

LLVM's code coverage mapping format is designed to be a self contained
data format, that can be embedded into the LLVM IR and object files.
It's described in this document as a **mapping** format because its goal is
to store the data that is required for a code coverage tool to map between
the specific source ranges in a file and the execution counts obtained
after running the instrumented version of the program.

The mapping data is used in two places in the code coverage process:

1. When clang compiles a source file with ``-fcoverage-mapping``, it
   generates the mapping information that describes the mapping between the
   source ranges and the profiling instrumentation counters.
   This information gets embedded into the LLVM IR and conveniently
   ends up in the final executable file when the program is linked.

2. It is also used by *llvm-cov* - the mapping information is extracted from an
   object file and is used to associate the execution counts (the values of the
   profile instrumentation counters), and the source ranges in a file.
   After that, the tool is able to generate various code coverage reports
   for the program.

The coverage mapping format aims to be a "universal format" that would be
suitable for usage by any frontend, and not just by Clang. It also aims to
provide the frontend the possibility of generating the minimal coverage mapping
data in order to reduce the size of the IR and object files - for example,
instead of emitting mapping information for each statement in a function, the
frontend is allowed to group the statements with the same execution count into
regions of code, and emit the mapping information only for those regions.

Advanced Concepts
=================

The remainder of this guide is meant to give you insight into the way the
coverage mapping format works.

The coverage mapping format operates on a per-function level as the
profile instrumentation counters are associated with a specific function.
For each function that requires code coverage, the frontend has to create
coverage mapping data that can map between the source code ranges and
the profile instrumentation counters for that function.

Mapping Region
--------------

The function's coverage mapping data contains an array of mapping regions.
A mapping region stores the `source code range`_ that is covered by this region,
the `file id <coverage file id_>`_, the `coverage mapping counter`_ and
the region's kind.
There are several kinds of mapping regions:

* Code regions associate portions of source code and `coverage mapping
  counters`_. They make up the majority of the mapping regions. They are used
  by the code coverage tool to compute the execution counts for lines,
  highlight the regions of code that were never executed, and to obtain
  the various code coverage statistics for a function.
  For example:

  :raw-html:`<pre class='highlight' style='line-height:initial;'><span>int main(int argc, const char *argv[]) </span><span style='background-color:#4A789C'>{    </span> <span class='c1'>// Code Region from 1:40 to 9:2</span>
  <span style='background-color:#4A789C'>                                            </span>
  <span style='background-color:#4A789C'>  if (argc &gt; 1) </span><span style='background-color:#85C1F5'>{                         </span>   <span class='c1'>// Code Region from 3:17 to 5:4</span>
  <span style='background-color:#85C1F5'>    printf("%s\n", argv[1]);              </span>
  <span style='background-color:#85C1F5'>  }</span><span style='background-color:#4A789C'> else </span><span style='background-color:#F6D55D'>{                                </span>   <span class='c1'>// Code Region from 5:10 to 7:4</span>
  <span style='background-color:#F6D55D'>    printf("\n");                         </span>
  <span style='background-color:#F6D55D'>  }</span><span style='background-color:#4A789C'>                                         </span>
  <span style='background-color:#4A789C'>  return 0;                                 </span>
  <span style='background-color:#4A789C'>}</span>
  </pre>`
* Skipped regions are used to represent source ranges that were skipped
  by Clang's preprocessor. They don't associate with
  `coverage mapping counters`_, as the frontend knows that they are never
  executed. They are used by the code coverage tool to mark the skipped lines
  inside a function as non-code lines that don't have execution counts.
  For example:

  :raw-html:`<pre class='highlight' style='line-height:initial;'><span>int main() </span><span style='background-color:#4A789C'>{               </span> <span class='c1'>// Code Region from 1:12 to 6:2</span>
  <span style='background-color:#85C1F5'>#ifdef DEBUG             </span>   <span class='c1'>// Skipped Region from 2:1 to 4:2</span>
  <span style='background-color:#85C1F5'>  printf("Hello world"); </span>
  <span style='background-color:#85C1F5'>#</span><span style='background-color:#4A789C'>endif                     </span>
  <span style='background-color:#4A789C'>  return 0;                </span>
  <span style='background-color:#4A789C'>}</span>
  </pre>`
* Expansion regions are used to represent Clang's macro expansions. They
  have an additional property - *expanded file id*. This property can be
  used by the code coverage tool to find the mapping regions that are created
  as a result of this macro expansion, by checking if their file id matches the
  expanded file id. They don't associate with `coverage mapping counters`_,
  as the code coverage tool can determine the execution count for this region
  by looking up the execution count of the first region with a corresponding
  file id.
  For example:

  :raw-html:`<pre class='highlight' style='line-height:initial;'><span>int func(int x) </span><span style='background-color:#4A789C'>{                             </span>
  <span style='background-color:#4A789C'>  #define MAX(x,y) </span><span style='background-color:#85C1F5'>((x) &gt; (y)? </span><span style='background-color:#F6D55D'>(x)</span><span style='background-color:#85C1F5'> : </span><span style='background-color:#F4BA70'>(y)</span><span style='background-color:#85C1F5'>)</span><span style='background-color:#4A789C'>     </span>
  <span style='background-color:#4A789C'>  return </span><span style='background-color:#7FCA9F'>MAX</span><span style='background-color:#4A789C'>(x, 42);                          </span> <span class='c1'>// Expansion Region from 3:10 to 3:13</span>
  <span style='background-color:#4A789C'>}</span>
  </pre>`

.. _source code range:

Source Range:
^^^^^^^^^^^^^

The source range record contains the starting and ending location of a certain
mapping region. Both locations include the line and the column numbers.

.. _coverage file id:

File ID:
^^^^^^^^

The file id an integer value that tells us
in which source file or macro expansion is this region located.
It enables Clang to produce mapping information for the code
defined inside macros, like this example demonstrates:

:raw-html:`<pre class='highlight' style='line-height:initial;'><span>void func(const char *str) </span><span style='background-color:#4A789C'>{        </span> <span class='c1'>// Code Region from 1:28 to 6:2 with file id 0</span>
<span style='background-color:#4A789C'>  #define PUT </span><span style='background-color:#85C1F5'>printf("%s\n", str)</span><span style='background-color:#4A789C'>   </span> <span class='c1'>// 2 Code Regions from 2:15 to 2:34 with file ids 1 and 2</span>
<span style='background-color:#4A789C'>  if(*str)                          </span>
<span style='background-color:#4A789C'>    </span><span style='background-color:#F6D55D'>PUT</span><span style='background-color:#4A789C'>;                            </span> <span class='c1'>// Expansion Region from 4:5 to 4:8 with file id 0 that expands a macro with file id 1</span>
<span style='background-color:#4A789C'>  </span><span style='background-color:#F6D55D'>PUT</span><span style='background-color:#4A789C'>;                              </span> <span class='c1'>// Expansion Region from 5:3 to 5:6 with file id 0 that expands a macro with file id 2</span>
<span style='background-color:#4A789C'>}</span>
</pre>`

.. _coverage mapping counter:
.. _coverage mapping counters:

Counter:
^^^^^^^^

A coverage mapping counter can represents a reference to the profile
instrumentation counter. The execution count for a region with such counter
is determined by looking up the value of the corresponding profile
instrumentation counter.

It can also represent a binary arithmetical expression that operates on
coverage mapping counters or other expressions.
The execution count for a region with an expression counter is determined by
evaluating the expression's arguments and then adding them together or
subtracting them from one another.
In the example below, a subtraction expression is used to compute the execution
count for the compound statement that follows the *else* keyword:

:raw-html:`<pre class='highlight' style='line-height:initial;'><span>int main(int argc, const char *argv[]) </span><span style='background-color:#4A789C'>{   </span> <span class='c1'>// Region's counter is a reference to the profile counter #0</span>
<span style='background-color:#4A789C'>                                           </span>
<span style='background-color:#4A789C'>  if (argc &gt; 1) </span><span style='background-color:#85C1F5'>{                        </span>   <span class='c1'>// Region's counter is a reference to the profile counter #1</span>
<span style='background-color:#85C1F5'>    printf("%s\n", argv[1]);             </span><span>   </span>
<span style='background-color:#85C1F5'>  }</span><span style='background-color:#4A789C'> else </span><span style='background-color:#F6D55D'>{                               </span>   <span class='c1'>// Region's counter is an expression (reference to the profile counter #0 - reference to the profile counter #1)</span>
<span style='background-color:#F6D55D'>    printf("\n");                        </span>
<span style='background-color:#F6D55D'>  }</span><span style='background-color:#4A789C'>                                        </span>
<span style='background-color:#4A789C'>  return 0;                                </span>
<span style='background-color:#4A789C'>}</span>
</pre>`

Finally, a coverage mapping counter can also represent an execution count of
of zero. The zero counter is used to provide coverage mapping for
unreachable statements and expressions, like in the example below:

:raw-html:`<pre class='highlight' style='line-height:initial;'><span>int main() </span><span style='background-color:#4A789C'>{                  </span>
<span style='background-color:#4A789C'>  return 0;                   </span>
<span style='background-color:#4A789C'>  </span><span style='background-color:#85C1F5'>printf("Hello world!\n")</span><span style='background-color:#4A789C'>;   </span> <span class='c1'>// Unreachable region's counter is zero</span>
<span style='background-color:#4A789C'>}</span>
</pre>`

The zero counters allow the code coverage tool to display proper line execution
counts for the unreachable lines and highlight the unreachable code.
Without them, the tool would think that those lines and regions were still
executed, as it doesn't possess the frontend's knowledge.

LLVM IR Representation
======================

The coverage mapping data is stored in the LLVM IR using a single global
constant structure variable called *__llvm_coverage_mapping*
with the *__llvm_covmap* section specifier.

For example, let’s consider a C file and how it gets compiled to LLVM:

.. _coverage mapping sample:

.. code-block:: c

  int foo() {
    return 42;
  }
  int bar() {
    return 13;
  }

The coverage mapping variable generated by Clang has 3 fields:

* Coverage mapping header.

* An array of function records.

* Coverage mapping data which is an array of bytes. Zero paddings are added at the end to force 8 byte alignment.

.. code-block:: llvm

  @__llvm_coverage_mapping = internal constant { { i32, i32, i32, i32 }, [2 x { i8*, i32, i32, i64 }], [40 x i8] }
  { 
    { i32, i32, i32, i32 } ; Coverage map header
    {
      i32 2,  ; The number of function records
      i32 20, ; The length of the string that contains the encoded translation unit filenames
      i32 20, ; The length of the string that contains the encoded coverage mapping data
      i32 0,  ; Coverage mapping format version
    },
    [2 x { i8*, i32, i32, i64 }] [ ; Function records
     { i8*, i32, i32, i64 } { i8* getelementptr inbounds ([3 x i8]* @__profn_foo, i32 0, i32 0), ; Function's name
       i32 3, ; Function's name length
       i32 9, ; Function's encoded coverage mapping data string length
       i64 0  ; Function's structural hash
     },
     { i8*, i32, i32, i64 } { i8* getelementptr inbounds ([3 x i8]* @__profn_bar, i32 0, i32 0), ; Function's name
       i32 3, ; Function's name length
       i32 9, ; Function's encoded coverage mapping data string length
       i64 0  ; Function's structural hash
     }],
   [40 x i8] c"..." ; Encoded data (dissected later)
  }, section "__llvm_covmap", align 8

Coverage Mapping Header:
------------------------

The coverage mapping header has the following fields:

* The number of function records.

* The length of the string in the third field of *__llvm_coverage_mapping* that contains the encoded translation unit filenames.

* The length of the string in the third field of *__llvm_coverage_mapping* that contains the encoded coverage mapping data.

* The format version. 0 is the first (current) version of the coverage mapping format.

.. _function records:

Function record:
----------------

A function record is a structure of the following type:

.. code-block:: llvm

  { i8*, i32, i32, i64 }

It contains the pointer to the function's name, function's name length,
the length of the encoded mapping data for that function, and function's 
hash value.

Encoded data:
-------------

The encoded data is stored in a single string that contains
the encoded filenames used by this translation unit and the encoded coverage
mapping data for each function in this translation unit.

The encoded data has the following structure:

``[filenames, coverageMappingDataForFunctionRecord0, coverageMappingDataForFunctionRecord1, ..., padding]``

If necessary, the encoded data is padded with zeroes so that the size
of the data string is rounded up to the nearest multiple of 8 bytes.

Dissecting the sample:
^^^^^^^^^^^^^^^^^^^^^^

Here's an overview of the encoded data that was stored in the
IR for the `coverage mapping sample`_ that was shown earlier:

* The IR contains the following string constant that represents the encoded
  coverage mapping data for the sample translation unit:

  .. code-block:: llvm

    c"\01\12/Users/alex/test.c\01\00\00\01\01\01\0C\02\02\01\00\00\01\01\04\0C\02\02\00\00"

* The string contains values that are encoded in the LEB128 format, which is
  used throughout for storing integers. It also contains a string value.

* The length of the substring that contains the encoded translation unit
  filenames is the value of the second field in the *__llvm_coverage_mapping*
  structure, which is 20, thus the filenames are encoded in this string:

  .. code-block:: llvm

    c"\01\12/Users/alex/test.c"

  This string contains the following data:

  * Its first byte has a value of ``0x01``. It stores the number of filenames
    contained in this string.
  * Its second byte stores the length of the first filename in this string.
  * The remaining 18 bytes are used to store the first filename.

* The length of the substring that contains the encoded coverage mapping data
  for the first function is the value of the third field in the first
  structure in an array of `function records`_ stored in the
  third field of the *__llvm_coverage_mapping* structure, which is the 9.
  Therefore, the coverage mapping for the first function record is encoded
  in this string:

  .. code-block:: llvm

    c"\01\00\00\01\01\01\0C\02\02"

  This string consists of the following bytes:

  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x01`` | The number of file ids used by this function. There is only one file id used by the mapping data in this function.      |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x00`` | An index into the filenames array which corresponds to the file "/Users/alex/test.c".                                   |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x00`` | The number of counter expressions used by this function. This function doesn't use any expressions.                     |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x01`` | The number of mapping regions that are stored in an array for the function's file id #0.                                |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x01`` | The coverage mapping counter for the first region in this function. The value of 1 tells us that it's a coverage        |
  |          | mapping counter that is a reference to the profile instrumentation counter with an index of 0.                          |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x01`` | The starting line of the first mapping region in this function.                                                         |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x0C`` | The starting column of the first mapping region in this function.                                                       |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x02`` | The ending line of the first mapping region in this function.                                                           |
  +----------+-------------------------------------------------------------------------------------------------------------------------+
  | ``0x02`` | The ending column of the first mapping region in this function.                                                         |
  +----------+-------------------------------------------------------------------------------------------------------------------------+

* The length of the substring that contains the encoded coverage mapping data
  for the second function record is also 9. It's structured like the mapping data
  for the first function record.

* The two trailing bytes are zeroes and are used to pad the coverage mapping
  data to give it the 8 byte alignment.

Encoding
========

The per-function coverage mapping data is encoded as a stream of bytes,
with a simple structure. The structure consists of the encoding
`types <cvmtypes_>`_ like variable-length unsigned integers, that
are used to encode `File ID Mapping`_, `Counter Expressions`_ and
the `Mapping Regions`_.

The format of the structure follows:

  ``[file id mapping, counter expressions, mapping regions]``

The translation unit filenames are encoded using the same encoding
`types <cvmtypes_>`_ as the per-function coverage mapping data, with the
following structure:

  ``[numFilenames : LEB128, filename0 : string, filename1 : string, ...]``

.. _cvmtypes:

Types
-----

This section describes the basic types that are used by the encoding format
and can appear after ``:`` in the ``[foo : type]`` description.

.. _LEB128:

LEB128
^^^^^^

LEB128 is an unsigned interger value that is encoded using DWARF's LEB128
encoding, optimizing for the case where values are small
(1 byte for values less than 128).

.. _Strings:

Strings
^^^^^^^

``[length : LEB128, characters...]``

String values are encoded with a `LEB value <LEB128_>`_ for the length
of the string and a sequence of bytes for its characters.

.. _file id mapping:

File ID Mapping
---------------

``[numIndices : LEB128, filenameIndex0 : LEB128, filenameIndex1 : LEB128, ...]``

File id mapping in a function's coverage mapping stream
contains the indices into the translation unit's filenames array.

Counter
-------

``[value : LEB128]``

A `coverage mapping counter`_ is stored in a single `LEB value <LEB128_>`_.
It is composed of two things --- the `tag <counter-tag_>`_
which is stored in the lowest 2 bits, and the `counter data`_ which is stored
in the remaining bits.

.. _counter-tag:

Tag:
^^^^

The counter's tag encodes the counter's kind
and, if the counter is an expression, the expression's kind.
The possible tag values are:

* 0 - The counter is zero.

* 1 - The counter is a reference to the profile instrumentation counter.

* 2 - The counter is a subtraction expression.

* 3 - The counter is an addition expression.

.. _counter data:

Data:
^^^^^

The counter's data is interpreted in the following manner:

* When the counter is a reference to the profile instrumentation counter,
  then the counter's data is the id of the profile counter.
* When the counter is an expression, then the counter's data
  is the index into the array of counter expressions.

.. _Counter Expressions:

Counter Expressions
-------------------

``[numExpressions : LEB128, expr0LHS : LEB128, expr0RHS : LEB128, expr1LHS : LEB128, expr1RHS : LEB128, ...]``

Counter expressions consist of two counters as they
represent binary arithmetic operations.
The expression's kind is determined from the `tag <counter-tag_>`_ of the
counter that references this expression.

.. _Mapping Regions:

Mapping Regions
---------------

``[numRegionArrays : LEB128, regionsForFile0, regionsForFile1, ...]``

The mapping regions are stored in an array of sub-arrays where every
region in a particular sub-array has the same file id.

The file id for a sub-array of regions is the index of that
sub-array in the main array e.g. The first sub-array will have the file id
of 0.

Sub-Array of Regions
^^^^^^^^^^^^^^^^^^^^

``[numRegions : LEB128, region0, region1, ...]``

The mapping regions for a specific file id are stored in an array that is
sorted in an ascending order by the region's starting location.

Mapping Region
^^^^^^^^^^^^^^

``[header, source range]``

The mapping region record contains two sub-records ---
the `header`_, which stores the counter and/or the region's kind,
and the `source range`_ that contains the starting and ending
location of this region.

.. _header:

Header
^^^^^^

``[counter]``

or

``[pseudo-counter]``

The header encodes the region's counter and the region's kind.

The value of the counter's tag distinguishes between the counters and
pseudo-counters --- if the tag is zero, than this header contains a
pseudo-counter, otherwise this header contains an ordinary counter.

Counter:
""""""""

A mapping region whose header has a counter with a non-zero tag is
a code region.

Pseudo-Counter:
"""""""""""""""

``[value : LEB128]``

A pseudo-counter is stored in a single `LEB value <LEB128_>`_, just like
the ordinary counter. It has the following interpretation:

* bits 0-1: tag, which is always 0.

* bit 2: expansionRegionTag. If this bit is set, then this mapping region
  is an expansion region.

* remaining bits: data. If this region is an expansion region, then the data
  contains the expanded file id of that region.

  Otherwise, the data contains the region's kind. The possible region
  kind values are:

  * 0 - This mapping region is a code region with a counter of zero.
  * 2 - This mapping region is a skipped region.

.. _source range:

Source Range
^^^^^^^^^^^^

``[deltaLineStart : LEB128, columnStart : LEB128, numLines : LEB128, columnEnd : LEB128]``

The source range record contains the following fields:

* *deltaLineStart*: The difference between the starting line of the
  current mapping region and the starting line of the previous mapping region.

  If the current mapping region is the first region in the current
  sub-array, then it stores the starting line of that region.

* *columnStart*: The starting column of the mapping region.

* *numLines*: The difference between the ending line and the starting line
  of the current mapping region.

* *columnEnd*: The ending column of the mapping region.
