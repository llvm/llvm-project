IR Tracker (IR TSV + SQLite query DB)
=====================================

.. contents::
   :local:

Overview
========

The **IR tracker** records how LLVM IR evolves through the new pass manager into a
compact tab-separated file. It can also record MIR snapshots from new-PM CodeGen
pipelines that use ``MachineFunction`` pass instrumentation. A small Python tool
can then post-process that TSV output into a SQLite database for indexed
queries. Each instruction row is tied to a tracker ID, which maps to a
``DILocation`` when one exists. If an input module has no debug information, the
recorder synthesizes locations first so the tool can still track IR evolution
through the pipeline.

Real source locations are only needed when you want to map tracked instructions
back to the original source file and line. For that workflow, compile with
``-g`` so the input IR carries source ``!dbg`` attachments.

Recording is enabled with the hidden LLVM option
``-ir-tracker-output=/absolute/path.tsv``. The hooks live in
``StandardInstrumentations`` and therefore apply to any tool that runs the new
pass manager with that instrumentation (``opt``, ``clang``, etc.). MIR tracking
is currently limited to new-PM CodeGen; default legacy-PM ``llc`` pipelines do
not emit MIR tracker rows.

The stream uses ``P`` rows for pass snapshots, ``T`` rows for tracker-ID source
locations, and ``I`` rows for instruction snapshots.

Recording with ``opt``
======================

.. code-block:: bash

  opt -disable-output -passes='default<O2>' \
    -ir-tracker-output=/tmp/pipeline.tsv input.ll

Use an absolute output path. The input IR does not need debug info for
pass-pipeline tracking; the recorder will synthesize locations for instructions
that need them. If you want queries to refer to the original source file and
line, produce the input IR with ``-g``.

Recording with ``clang``
========================

Forward the option through Clang with ``-mllvm`` so the middle-end sees the same
flag as ``opt``:

.. code-block:: bash

  clang -O1 -emit-llvm -S -g sum.c -o sum.ll \
    -mllvm -ir-tracker-output=/tmp/pipeline.tsv

Here ``-g`` is optional for tracking the IR itself, but it preserves source
locations so later ``trace`` / ``show`` queries can use source file and line
numbers. ``-O1`` (or another ``-O`` level) selects the usual optimization
pipeline that ``opt`` would run for that tier.

Recording MIR with new-PM ``llc``
=================================

MIR tracking uses the same hidden option, but requires the new CodeGen pass
manager:

.. code-block:: bash

  llc -enable-new-pm -filetype=null -ir-tracker-output=/tmp/pipeline.tsv input.ll

The final MIR snapshots are often close enough to assembly to debug instruction
selection, register allocation, spills, and late machine optimizations. The
tracker does not record final assembly text or MC streamer directives.

SQLite build step
=================

The Python driver can convert the TSV output into a SQLite database:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py build \
    --input /tmp/pipeline.tsv --db /tmp/pipeline.db

The resulting database uses ``schema_version = 2`` in ``ir_tracker_meta``. The
main tables are:

* ``ir_tracker_meta`` — key/value metadata (including ``schema_version``)
* ``ir_tracker_files`` — deduplicated paths from ``DIFile`` (often a basename
  such as ``sum.c``)
* ``ir_tracker_passes`` — one row per snapshot: ``seq``, ``kind`` (``ir`` or
  ``mir``), ``phase`` (``initial`` or ``after``), ``pass_class``, ``ir_unit``
* ``ir_tracker_instructions`` — instruction text and opcode per pass, keyed by
  ``file_id``, ``line``, ``col``

Query tool
==========

The Python driver lives at ``llvm/tools/ir-tracker/ir-tracker.py`` (installed
under ``<prefix>/share/ir-tracker/`` when the ``ir-tracker`` install component
is enabled). It can build the SQLite DB from tracker TSV output and then query
that DB. Subcommands:

* ``build`` — convert tracker TSV output into a SQLite database
* ``passes`` — list recorded passes in ``seq`` order; use ``--kind ir``,
  ``--kind mir``, or ``--kind all`` to filter representations
* ``trace`` — summarize the first and last pass that still have instructions
  matching a source location; defaults to ``--kind ir``
* ``show`` — print the instructions matching ``--file`` / ``--line`` (and
  optional ``--col`` / ``--opcode``) across passes; by default only passes where
  the printed IR **changed** are shown; use ``--kind mir`` for MIR rows,
  ``--all-passes`` for every pass, or ``--seq N`` for one pass
* ``html`` — generate a static HTML report with one page per function
* ``sql`` — run a single read-only SQL statement

The ``--file`` argument is matched against the path stored in
``ir_tracker_files`` (substring match, case-insensitive). Clang usually records
the ``DIFile`` basename, so prefer ``--file sum.c`` rather than a full host path.

HTML report
===========

The ``html`` subcommand generates a static report directory from a built query
database:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py html \
    --db /tmp/pipeline.db -o /tmp/pipeline-html

The output contains ``index.html``, ``style.css``, and one ``fn-*.html`` page
per function. Each function page has four panels:

* a function list grouped by file
* the initial IR snapshot for the selected function
* the final IR snapshot, when the recorder emitted a ``phase='final'`` snapshot
* pass history for the selected instruction location

Clicking an instruction in the initial or final panel shows the pass-by-pass
history for the same tracked location. Rows that share a tracked location are
highlighted with the same background color across panels.

Useful options:

* ``--file TEXT`` — only emit pages for source paths containing ``TEXT``

The current function-centric report is generated entirely from the database and
does not read source files. ``--source-dir``, ``--all-passes``, and
``--no-highlight`` are accepted for command-line consistency with related report
generators, but they do not change this layout.

Example: following one source line through ``clang -O1``
========================================================

Source file ``sum.c``:

.. code-block:: c

  /* Example: trivial fold (x + 0) -> x */
  int bump(int x) {
    return x + 0;
  }

Recording (same command as in *Recording with ``clang``*):

.. code-block:: bash

  clang -O1 -emit-llvm -S -g sum.c -o sum.ll \
    -mllvm -ir-tracker-output=/tmp/pipeline.tsv

Then build the query database:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py build \
    --input /tmp/pipeline.tsv --db /tmp/pipeline.db

The following excerpts come from a real ``ir-tracker`` run against the database
produced that way. **Pass names and sequence numbers depend on your Clang/LLVM
version, target, and optimization level**; treat pass sequence numbers as
illustrative, not a stable ABI.

List passes (truncated):

.. code-block:: text

      0  id=1       initial  '<initial>'  on '[module]'
      1  id=2       after  'memprof-remove-attributes'  on '[module]'
      2  id=3       after  'annotation2metadata'  on '[module]'
      …
     10  id=11      after  'sroa'  on 'bump'
     11  id=12      after  'early-cse'  on 'bump'
     …

Trace line ``3`` (the ``return x + 0;`` line in ``sum.c``):

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py trace \
    --db /tmp/pipeline.db --file sum.c --line 3

.. code-block:: text

  Matches at final pass (seq=94): 1 instruction(s)
  First pass with any matching instruction: seq=0 <initial> on [module] (3 row(s))

``show`` without ``--all-passes`` prints only passes where the matched IR text
changed: here the load/add/return cluster simplifies until ``early-cse`` folds
``x + 0`` to ``x``:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py show \
    --db /tmp/pipeline.db --file sum.c --line 3

.. code-block:: text

  seq=0 '<initial>' on '[module]'
    function bump, block entry:
        %0 = load i32, ptr %x.addr, align 4
        %add = add nsw i32 %0, 0
        ret i32 %add
  seq=10 'sroa' on 'bump'
    function bump, block entry:
        %add = add nsw i32 %x, 0
        ret i32 %add
  seq=11 'early-cse' on 'bump'
    function bump, block entry:
        ret i32 %x

The initial snapshot for the same line (``--seq 0``) recovers the unoptimized
cluster before any pass runs:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py show \
    --db /tmp/pipeline.db --file sum.c --line 3 --seq 0

.. code-block:: text

  seq=0 '<initial>' on '[module]'
    function bump, block entry:
        %0 = load i32, ptr %x.addr, align 4
        %add = add nsw i32 %0, 0
        ret i32 %add

Tests
=====

* Recorder: ``llvm/test/Other/ir-tracker-db.ll``
* Query tool: ``llvm/test/tools/llvm-ir-tracker/``

Limitations
===========

* **IR only** — there is no MIR, object, or assembly capture in this schema.
* **Source attribution needs source locations** — the tracker can follow IR
  evolution without debug info by synthesizing locations, but those synthetic
  locations do not identify original source files and lines.
* **Locations are keys, not proofs** — optimizations can merge, clone, or drop
  instructions; the database lists what survived each pass with a given
  location, not a formal def-use proof.
