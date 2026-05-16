llvm-advisor - LLVM compilation snapshot and analysis tool
===========================================================

.. program:: llvm-advisor

SYNOPSIS
--------

:program:`llvm-advisor` <command> [*options*]

DESCRIPTION
-----------

The :program:`llvm-advisor` tool is a compilation analysis utility that captures
and analyzes compilation snapshots from LLVM-based builds. It uses ``compile_commands.json``
to identify compilation units, captures detailed analysis data using configurable
capabilities, and provides tools for comparing, querying, and inspecting results.

:program:`llvm-advisor` operates on the concept of **snapshots** - immutable
records of a compilation captured at a point in time. Each snapshot contains:

* LLVM IR and assembly output
* Optimization remarks and analysis data
* Compiler diagnostics and warnings
* Debug information
* Runtime profiling data (when correlated)
* Results from multiple analysis capabilities

The captured data is stored in a content-addressable storage system and can be
queried, compared, and analyzed using the CLI or HTTP server interface.

GLOBAL OPTIONS
--------------

.. option:: --store <directory>

 Specify the advisor store directory where snapshots and analysis data are
 stored. Defaults to platform-specific location (``~/.local/share/llvm-advisor``
 on Linux, ``~/Library/Application Support/llvm-advisor`` on macOS).

.. option:: --capability-dir <directory>

 Directory containing capability JSON specifications. Defaults to
 ``config/capabilities`` relative to the advisor installation.

.. option:: --plugin <path>

 Load an external advisor plugin (.so/.dylib/.dll). May be repeated to load
 multiple plugins. Plugins can register custom capabilities at runtime.

.. option:: --help, -h

 Display usage information and available commands.

COMMANDS
--------

:program:`llvm-advisor` supports the following subcommands:

**capture**
~~~~~~~~~~~

Create a snapshot of a build based on ``compile_commands.json``:

.. code-block:: console

  llvm-advisor capture [--source-root <dir>] [--build-root <dir>]
                       [--profile <profile.json>]
                       [--capability <id>...]

Captures compilation data for all units in ``compile_commands.json``. The tool
auto-detects source and build roots if not specified. Capabilities can be
specified via profile file or individual ``--capability`` flags.

**list**
~~~~~~~

List snapshots or compilation units:

.. code-block:: console

  llvm-advisor list [--snapshot <id>]

Without options, lists all snapshots. With ``--snapshot``, lists units in that
snapshot. Use ``latest`` to refer to the most recent snapshot.

**query**
~~~~~~~~~

Run capabilities for a unit or snapshot:

.. code-block:: console

  llvm-advisor query --snapshot <id> --capability <id>...
                    [--unit <id>]

Executes specified capabilities and outputs results as JSON. Can target a
specific unit or the entire snapshot.

**inspect**
~~~~~~~~~~~

Inspect and display capability results:

.. code-block:: console

  llvm-advisor inspect [<mode>] --snapshot <id>
                       [--unit <id>] [--capability <id>]
                       [--file <path>] [--line <num>]
                       [--output-format json|text]

Displays capability results in human-readable or JSON format. Supports filtering
by file, line, and other criteria. Available modes: ``ir``, ``ir-diff``, ``cfg``,
``dom``, ``callgraph``, ``dag``, and others.

**compare**
~~~~~~~~~~~

Compare two snapshots:

.. code-block:: console

  llvm-advisor compare --before <id> --after <id>

Generates a detailed comparison report between two snapshots, highlighting
differences in optimization, performance metrics, and diagnostic data.

**serve**
~~~~~~~~~

Run the embedded HTTP server:

.. code-block:: console

  llvm-advisor serve [--port <port>]

Launches an HTTP server (default port 8080) providing REST API access to all
snapshot data and analysis results.

**capabilities**
~~~~~~~~~~~~~~~~

List available capabilities:

.. code-block:: console

  llvm-advisor capabilities

Displays all registered capabilities and their metadata.

**runtime-ingest**
~~~~~~~~~~~~~~~~~~~

Ingest runtime execution data:

.. code-block:: console

  llvm-advisor runtime-ingest --snapshot <id>
                              --kind <type>
                              --data <path>

Ingests runtime data (PGO, coverage, memprof, etc.) into a snapshot for
correlation and analysis.

**runtime-correlate**
~~~~~~~~~~~~~~~~~~~~~

Correlate runtime data with captured units:

.. code-block:: console

  llvm-advisor runtime-correlate --snapshot <id>

Links ingested runtime data to specific compilation units.

**insight-list**
~~~~~~~~~~~~~~~~

List available insights:

.. code-block:: console

  llvm-advisor insight-list [--snapshot <id>] [--unit <id>]

Displays available insight analyses for the given context.

**insight**
~~~~~~~~~~~

Run insight analysis:

.. code-block:: console

  llvm-advisor insight --name <name> --snapshot <id>
                       [--unit <id>] [--baseline <id>]

Executes a named insight analysis and returns results.

**health**
~~~~~~~~~~

Check service health:

.. code-block:: console

  llvm-advisor health

Verifies the advisor service is operational.

**inspect-storage**
~~~~~~~~~~~~~~~~~~~

Inspect storage state:

.. code-block:: console

  llvm-advisor inspect-storage

Displays information about the content-addressable storage backend.

**maintenance-compact**
~~~~~~~~~~~~~~~~~~~~~~~

Compact CAS storage:

.. code-block:: console

  llvm-advisor maintenance-compact

Performs maintenance operations on the storage system to reclaim space.

EXAMPLES
--------

Basic Snapshot Capture
~~~~~~~~~~~~~~~~~~~~~~

Capture a snapshot from a CMake project:

.. code-block:: console

  cd /path/to/project
  llvm-advisor capture

The tool automatically detects ``compile_commands.json`` and captures all
compilation units using default capabilities.

Capture with Specific Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Capture specific analyses:

.. code-block:: console

  llvm-advisor capture --capability llvm.ir.view --capability llvm.mca.report

This captures only IR and machine code analysis results.

Using a Profile
~~~~~~~~~~~~~~~

Capture with a predefined profile:

.. code-block:: console

  llvm-advisor capture --profile config/capabilities/catalog.json

Load capability specifications from a profile file.

List All Snapshots
~~~~~~~~~~~~~~~~~~

.. code-block:: console

  llvm-advisor list

Shows all captured snapshots with their IDs and metadata.

List Units in a Snapshot
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

  llvm-advisor list --snapshot latest

Lists all compilation units in the most recent snapshot.

Query a Snapshot
~~~~~~~~~~~~~~~~

Run capabilities and get results as JSON:

.. code-block:: console

  llvm-advisor query --snapshot latest --capability llvm.ir.view

Outputs JSON results of the specified capability.

Inspect IR Output
~~~~~~~~~~~~~~~~~

View LLVM IR for a specific unit:

.. code-block:: console

  llvm-advisor inspect ir --snapshot latest --unit main.cpp

Compare Two Snapshots
~~~~~~~~~~~~~~~~~~~~~

Generate a detailed comparison:

.. code-block:: console

  llvm-advisor compare --before snapshot-1 --after snapshot-2

Displays differences in optimization and diagnostics between snapshots.

Run the Web Server
~~~~~~~~~~~~~~~~~~

Start the HTTP interface:

.. code-block:: console

  llvm-advisor serve --port 8080

Access analysis at ``http://localhost:8080``.

Ingest PGO Data
~~~~~~~~~~~~~~~

Add profiling information to a snapshot:

.. code-block:: console

  llvm-advisor runtime-ingest --snapshot latest --kind pgo-instr --data profile.profraw
  llvm-advisor runtime-correlate --snapshot latest

Links profiling data to compilation units for optimization analysis.

CONFIGURATION
-------------

:program:`llvm-advisor` can be configured through environment variables and
command-line options.

**Environment Variables**

``LLVM_ADVISOR_STORE``
  Override the default store directory.

**Capability Configuration**

Capabilities are defined in JSON format in the capability directory
(``config/capabilities/catalog.json``). Each capability specifies:

* Capability ID (e.g., ``llvm.ir.view``)
* Name and description
* Required parameters
* Output format
* Readiness status (experimental, beta, stable)

**Custom Profiles**

You can create custom capability profiles as JSON files:

.. code-block:: json

  {
    "capabilities": [
      "llvm.ir.view",
      "llvm.mca.report",
      "llvm.cfg"
    ]
  }

Then reference with ``--profile custom-profile.json`` during capture.

OUTPUT FORMAT
-------------

:program:`llvm-advisor` stores data in a content-addressable storage system
under the store directory. Data is organized by snapshot and accessed through
capability results.

**Query Output**

When running queries or inspections, output is typically JSON:

.. code-block:: json

  {
    "capability": "llvm.ir.view",
    "unit": "main.cpp",
    "success": true,
    "value": {
      "ir": "; ... LLVM IR content ..."
    }
  }

**Compare Output**

Comparison results include changes, metrics, and diffs:

.. code-block:: json

  {
    "baseline": {
      "snapshot": "id-1",
      "value": {...}
    },
    "candidate": {
      "snapshot": "id-2",
      "value": {...}
    },
    "diff": {
      "changed": true,
      "changes": [...]
    }
  }

**Snapshot Metadata**

Each snapshot contains:

* Snapshot ID and creation timestamp
* List of compilation units captured
* Capabilities that were run
* Runtime data if correlated
* Storage hashes for content integrity

EXIT STATUS
-----------

:program:`llvm-advisor` returns 0 on success and non-zero on failure.

Exit code 1 indicates:

* Invalid command line arguments
* Missing or invalid snapshot ID
* Capability execution failure
* Storage access errors
* Malformed input data
* Service unavailable errors

SEE ALSO
--------

:manpage:`clang(1)`, :manpage:`opt(1)`, :manpage:`llc(1)`
