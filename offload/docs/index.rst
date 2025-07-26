Welcome to Offload's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :hidden:

   offload-api

Summary
-------

The Offload subproject aims at providing tooling, runtimes, and APIs that allow
users to execute code on accelerators or other "co-processors" that may or may
not match the architecture of their "host". In the long run, all kinds of
targets are in scope of this effort, including but not limited to: CPUs, GPUs,
FPGAs, AI/ML accelerators, distributed resources, etc.

For OpenMP offload users, the project is ready and fully usable. The final API
design is still under development. More content will show up here and on our
webpage soon. In the meantime, people are encouraged to participate in our
meetings (see below) and check our `development board
<https://github.com/orgs/llvm/projects/24/>`_ as well as the discussions on
`Discourse <https://discourse.llvm.org/tag/offload>`_.

Meetings
--------

Every second Wednesday, 7:00 - 8:00am PT, starting Jan 24, 2024. Alternates
with the OpenMP in LLVM meeting. `invite.ics
<https://drive.google.com/file/d/1AYwKdnM01aV9Gv9k435ArEAhn7PAer7z/view?usp=sharing>`_
`Meeting Minutes and Agenda
<https://docs.google.com/document/d/1PAeEshxHCv22JDBCPA9GXGggLp0t7rsnD_jL04MBbzw/edit?usp=sharing>`_.


Building
--------

A minimal Linux build CMake configuration:

.. code-block:: console

  $ cmake llvm -Bbuild \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;lldb;lld' \
      -DLLVM_ENABLE_RUNTIMES='offload;openmp'
  $ cmake --build build

* ``LLVM_ENABLE_RUNTIMES`` must include ``openmp`` as it is currently a
  dependency of ``offload`` during the initial transitional phase of the
  project.

.. hint::

  As part of the main build an `ExternalProject
  <https://cmake.org/cmake/help/latest/module/ExternalProject.html>`_ will be
  created at ``build/runtimes/runtimes-bins`` which contains the Offload
  sub-build. Additional build targets are present in the sub-build which are
  not accessible in the LLVM build.

Running Tests
^^^^^^^^^^^^^

There are two main check targets:

* ``check-offload`` runs the OpenMP tests, this is available in both the LLVM
  build directory as well as the runtimes-bin sub-build directory.
* ``check-offload-unit`` runs the Offload API unit test, this is only available
  in the runtimes-bin sub-build directory.

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

Additional CMake options are necessary to build the Sphinx documentation.
Firstly, we need to ensure the Python dependencies are available, ideally using
a virtual environment:

.. code-block:: console

  $ python -m venv env
  $ source env/bin/activate
  $ pip install -r llvm/docs/requirements.txt

Assuming we already have an existing build described above, we need to
reconfigure the Offload sub-build, this will enable the ``docs-offload-html``
target.

.. code-block:: console

  $ cmake runtimes -Bbuild/runtimes/runtimes-bins \
      -DLLVM_ENABLE_SPHINX=ON
  $ cmake --build build

Once the documentation is built it can be viewed on `localhost:8000
<http://localhost:8000>`_ as follows:

.. code-block:: console

  $ cd build/runtimes/runtimes-bins/offload/docs/html
  $ python -m http.server

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
