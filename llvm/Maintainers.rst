================
LLVM Maintainers
================

This file is a list of the
`maintainers <https://llvm.org/docs/DeveloperPolicy.html#maintainers>`_ for
LLVM.

.. contents::
   :depth: 2
   :local:

Current Maintainers
===================
The following people are the active code owners for the project. Please reach
out to them for code reviews, questions about their area of expertise, or other
assistance.

**Warning: The maintainer list for LLVM is currently not up to date.**

Lead maintainer
---------------
The lead maintainer is responsible for all parts of LLVM not covered by somebody else.

| Chris Lattner
| sabre\@nondot.org (email), lattner (GitHub), clattner (Discourse)



Transforms and analyses
-----------------------

AliasAnalysis
~~~~~~~~~~~~~~
| Hal Finkel
| hfinkel\@anl.gov (email), hfinkel (GitHub)

Attributor, OpenMPOpt
~~~~~~~~~~~~~~~~~~~~~
| Johannes Doerfert
| jdoerfert\@llnl.gov (email), jdoerfert (GitHub)

InferAddressSpaces
~~~~~~~~~~~~~~~~~~
| Matt Arsenault
| Matthew.Arsenault\@amd.com, arsenm2\@gmail.com (email), arsenm (GitHub)

Inlining
~~~~~~~~
| Chandler Carruth
| chandlerc\@gmail.com, chandlerc\@google.com (email), chandlerc (GitHub)

InstCombine, ConstantFold
~~~~~~~~~~~~~~~~~~~~~~~~~
| David Majnemer
| david.majnemer\@gmail.com (email), majnemer (GitHub)

InstrProfiling and related parts of ProfileData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Justin Bogner
| mail\@justinbogner.com (email), bogner (GitHub)

SampleProfile and related parts of ProfileData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Diego Novillo
| dnovillo\@google.com (email), dnovillo (GitHub)

LoopStrengthReduce
~~~~~~~~~~~~~~~~~~
| Quentin Colombet
| quentin.colombet\@gmail.com (email), qcolombet (GitHub)

LoopVectorize
~~~~~~~~~~~~~
| Florian Hahn
| flo\@fhahn.com (email), fhahn (GitHub)

ScalarEvolution, IndVarSimplify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Philip Reames
| listmail\@philipreames.com (email), preames (GitHub)

SLPVectorizer
~~~~~~~~~~~~~
| Alexey Bataev
| a.bataev\@outlook.com (email), alexey-bataev (GitHub)

SROA, Mem2Reg
~~~~~~~~~~~~~
| Chandler Carruth
| chandlerc\@gmail.com, chandlerc\@google.com (email), chandlerc (GitHub)



Instrumentation and sanitizers
------------------------------

AddressSanitizer, ThreadSanitizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Kostya Serebryany
| kcc\@google.com (email), kcc (GitHub)

MemorySanitizer
~~~~~~~~~~~~~~~
| Evgeniy Stepanov
| eugenis\@google.com (email), eugenis (GitHub)

RealtimeSanitizer
~~~~~~~~~~~~~~~~~
| Christopher Apple
| cja-private\@pm.me (email), cjappl (GitHub)
| David Trevelyan
| david.trevelyan\@gmail.com (email), davidtrevelyan (GitHub)



Generic backend and code generation
-----------------------------------

Parts of code generator not covered by someone else
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Evan Cheng
| evan.cheng\@apple.com (email)

SelectionDAG
~~~~~~~~~~~~
| Justin Bogner
| mail\@justinbogner.com (email), bogner (GitHub)

FastISel
~~~~~~~~
| Chad Rosier
| mcrosier\@codeaurora.org (email)

Instruction scheduling
~~~~~~~~~~~~~~~~~~~~~~
| Matthias Braun
| matze\@braunis.de (email), MatzeB (GitHub)

VLIW Instruction Scheduling, Packetization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Sergei Larin
| slarin\@codeaurora.org (email)

Register allocation
~~~~~~~~~~~~~~~~~~~
| Quentin Colombet
| quentin.colombet\@gmail.com (email), qcolombet (GitHub)

MC layer
~~~~~~~~
| James Grosbach
| grosbach\@apple.com (email)

Windows codegen
~~~~~~~~~~~~~~~
| Anton Korobeynikov
| anton\@korobeynikov.info (email), asl (GitHub)


Backends / Targets
------------------

AArch64 backend
~~~~~~~~~~~~~~~
| Tim Northover
| t.p.northover\@gmail.com (email), TNorthover (GitHub)

AMDGPU backend
~~~~~~~~~~~~~~
| Matt Arsenault
| Matthew.Arsenault\@amd.com, arsenm2\@gmail.com (email), arsenm (GitHub)

ARC backend
~~~~~~~~~~~
| Mark Schimmel
| marksl\@synopsys.com (email), markschimmel (GitHub)

ARM backend
~~~~~~~~~~~
| Renato Golin
| rengolin\@systemcall.eu (email), rengolin (GitHub)

AVR backend
~~~~~~~~~~~
| Ben Shi
| 2283975856\@qq.com, powerman1st\@163.com (email), benshi001 (GitHub)

BPF backend
~~~~~~~~~~~
| Alexei Starovoitov
| alexei.starovoitov\@gmail.com (email), 4ast (GitHub)

CSKY backend
~~~~~~~~~~~~
| Zi Xuan Wu (Zeson)
| zixuan.wu\@linux.alibaba.com (email), zixuan-wu (GitHub)

Hexagon backend
~~~~~~~~~~~~~~~
| Sundeep Kushwaha
| sundeepk\@quicinc.com (email)

Lanai backend
~~~~~~~~~~~~~
| Jacques Pienaar
| jpienaar\@google.com (email), jpienaar (GitHub)

LoongArch backend
~~~~~~~~~~~~~~~~~
| Weining Lu
| luweining\@loongson.cn (email), SixWeining (GitHub)

M68k backend
~~~~~~~~~~~~
| Min-Yih Hsu
| min\@myhsu.dev (email), mshockwave (GitHub)

MSP430 backend
~~~~~~~~~~~~~~
| Anton Korobeynikov
| anton\@korobeynikov.info (email), asl (GitHub)

NVPTX backend
~~~~~~~~~~~~~
| Justin Holewinski
| jholewinski\@nvidia.com (email), jholewinski (GitHub)

PowerPC backend
~~~~~~~~~~~~~~~
| Zheng Chen
| czhengsz\@cn.ibm.com (email), chenzheng1030 (GitHub)

RISCV backend
~~~~~~~~~~~~~
| Alex Bradbury
| asb\@igalia.com (email), asb (GitHub)

Sparc backend
~~~~~~~~~~~~~
| Venkatraman Govindaraju
| venkatra\@cs.wisc.edu (email), vegovin (GitHub)

SPIRV backend
~~~~~~~~~~~~~
| Ilia Diachkov
| ilia.diachkov\@gmail.com (email), iliya-diyachkov (GitHub)

SystemZ backend
~~~~~~~~~~~~~~~
| Ulrich Weigand
| uweigand\@de.ibm.com (email), uweigand (GitHub)

VE backend
~~~~~~~~~~
| Kazushi Marukawa
| marukawa\@nec.com (email), kaz7 (GitHub)

WebAssembly backend
~~~~~~~~~~~~~~~~~~~
| Dan Gohman
| llvm\@sunfishcode.online (email), sunfishcode (GitHub)

X86 backend
~~~~~~~~~~~
| Simon Pilgrim
| llvm-dev\@redking.me.uk (email), RKSimon (GitHub)
| Phoebe Wang
| phoebe.wang\@intel.com (email), phoebewang (GitHub)

XCore backend
~~~~~~~~~~~~~
| Nigel Perks
| nigelp\@xmos.com (email), nigelp-xmos (GitHub)

Xtensa backend
~~~~~~~~~~~~~~
| Andrei Safronov
| andrei.safronov\@espressif.com (email), andreisfr (GitHub)



Libraries and shared infrastructure
-----------------------------------

ADT, Support
~~~~~~~~~~~~
| Chandler Carruth
| chandlerc\@gmail.com, chandlerc\@google.com (email), chandlerc (GitHub)

Bitcode
~~~~~~~
| Peter Collingbourne
| peter\@pcc.me.uk (email), pcc (GitHub)

CMake and library layering
~~~~~~~~~~~~~~~~~~~~~~~~~~
| Chandler Carruth
| chandlerc\@gmail.com, chandlerc\@google.com (email), chandlerc (GitHub)

Debug info
~~~~~~~~~~
| Eric Christopher
| echristo\@gmail.com (email), echristo (GitHub)

DWARF Parser
~~~~~~~~~~~~
| Benjamin Kramer
| benny.kra\@gmail.com (email), d0k (GitHub)

IR Linker
~~~~~~~~~
| Teresa Johnson
| tejohnson\@google.com (email), teresajohnson (GitHub)

LTO
~~~
| Peter Collingbourne
| peter\@pcc.me.uk (email), pcc (GitHub)

MCJIT, Orc, RuntimeDyld, PerfJITEvents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Lang Hames
| lhames\@gmail.com (email), lhames (GitHub)

TableGen
~~~~~~~~
| Paul C. Anagnostopoulos
| paul\@windfall.com (email)

TextAPI
~~~~~~~
| Cyndy Ishida
| cyndyishida\@gmail.com (email), cyndyishida (GitHub)



Tools
-----

llvm-mca and MCA library
~~~~~~~~~~~~~~~~~~~~~~~~
| Andrea Di Biagio
| andrea.dibiagio\@sony.com, andrea.dibiagio\@gmail.com (email), adibiagio (GitHub)

llvm-objcopy and ObjCopy library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Jake Ehrlich
| jakehehrlich\@google.com (email), jakehehrlich (GitHub)

Gold plugin
~~~~~~~~~~~
| Teresa Johnson
| tejohnson\@google.com (email), teresajohnson (GitHub)


Other
-----

Release management
~~~~~~~~~~~~~~~~~~

For x.y.0 releases:

| Hans Wennborg
| hans\@chromium.org (email), zmodem (GitHub)
|

For x.y.[1-9] releases:

| Tom Stellard
| tstellar\@redhat.com (email), tstellar (GitHub)

MinGW support
~~~~~~~~~~~~~
| Martin Storsjö
| martin\@martin.st (email), mstrorsjo (GitHub)

Windows support in object tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Michael Spencer
| bigcheesegs\@gmail.com (email), Bigcheese (GitHub)

Sony PlayStation®4 support
~~~~~~~~~~~~~~~~~~~~~~~~~~
| Paul Robinson
| paul.robinson\@sony.com (email), pogo59 (GitHub)

Inline assembly
~~~~~~~~~~~~~~~
| Eric Christopher
| echristo\@gmail.com (email), echristo (GitHub)

Exception handling
~~~~~~~~~~~~~~~~~~
| Anton Korobeynikov
| anton\@korobeynikov.info (email), asl (GitHub)

ARM EABI
~~~~~~~~
| Anton Korobeynikov
| anton\@korobeynikov.info (email), asl (GitHub)

LLVM Buildbot
~~~~~~~~~~~~~
| Galina Kistanova
| gkistanova\@gmail.com (email), gkistanova (GitHub)



Other subprojects
-----------------

Some subprojects maintain their own list of per-component maintainers.
Others only have a lead maintainer listed here.

`Bolt maintainers <https://github.com/llvm/llvm-project/blob/main/bolt/CODE_OWNERS.TXT>`_

`Clang maintainers <https://github.com/llvm/llvm-project/blob/main/clang/CodeOwners.rst>`_

`Clang-tools-extra maintainers <https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/CODE_OWNERS.TXT>`_

`Compiler-rt maintainers <https://github.com/llvm/llvm-project/blob/main/compiler-rt/CODE_OWNERS.TXT>`_

`Flang maintainers <https://github.com/llvm/llvm-project/blob/main/flang/CODE_OWNERS.TXT>`_

`LLD maintainers <https://github.com/llvm/llvm-project/blob/main/lld/CODE_OWNERS.TXT>`_

`LLDB maintainers <https://github.com/llvm/llvm-project/blob/main/lldb/CodeOwners.rst>`_

libc++
~~~~~~
| Louis Dionne
| ldionne.2\@gmail.com (email), ldionne (GitHub)

libclc
~~~~~~
| Tom Stellard
| tstellar\@redhat.com (email), tstellar (GitHub)

OpenMP (runtime library)
~~~~~~~~~~~~~~~~~~~~~~~~
| Andrey Churbanov
| andrey.churbanov\@intel.com (email), AndreyChurbanov (GitHub)

Polly
~~~~~
| Tobias Grosser
| tobias\@grosser.es (email), tobiasgrosser (GitHub)


Inactive Maintainers
====================
The following people have graciously spent time performing maintainer
responsibilities but are no longer active in that role. Thank you for all your
help with the success of the project!

Emeritus lead maintainers
-------------------------

Inactive component maintainers
------------------------------

Former maintainers of removed components
----------------------------------------
| Duncan Sands (baldrick\@free.fr, CunningBaldrick) -- DragonEgg
| Hal Finkel (hfinkel\@anl.gov, hfinkel) -- LoopReroll
