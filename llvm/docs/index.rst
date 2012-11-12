.. _contents:

Overview
========

.. warning::

   If you are using a released version of LLVM, see `the download page
   <http://llvm.org/releases/>`_ to find your documentation.

The LLVM compiler infrastructure supports a wide range of projects, from
industrial strength compilers to specialized JIT applications to small
research projects.

Similarly, documentation is broken down into several high-level groupings
targeted at different audiences:

* **Design & Overview**

 Several introductory papers and presentations are available at
 :ref:`design_and_overview`.

* **Publications**

 The list of `publications <http://llvm.org/pubs>`_ based on LLVM.

* **User Guides**

 Those new to the LLVM system should first visit the :ref:`userguides`.

 NOTE: If you are a user who is only interested in using LLVM-based
 compilers, you should look into `Clang <http://clang.llvm.org>`_ or
 `DragonEgg <http://dragonegg.llvm.org>`_ instead. The documentation here is
 intended for users who have a need to work with the intermediate LLVM
 representation.

* **API Clients**

 Developers of applications which use LLVM as a library should visit the
 :ref:`programming`.

* **Subsystems**

 API clients and LLVM developers may be interested in the
 :ref:`subsystems` documentation.

* **Development Process**

 Additional documentation on the LLVM project can be found at
 :ref:`development_process`.

* **Mailing Lists**

 For more information, consider consulting the LLVM :ref:`mailing_lists`.

.. toctree::
   :maxdepth: 2

   design_and_overview
   userguides
   programming
   subsystems
   development_process
   mailing_lists
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
