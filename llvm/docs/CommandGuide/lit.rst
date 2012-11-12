lit - LLVM Integrated Tester
============================


SYNOPSIS
--------


**lit** [*options*] [*tests*]


DESCRIPTION
-----------


**lit** is a portable tool for executing LLVM and Clang style test suites,
summarizing their results, and providing indication of failures. **lit** is
designed to be a lightweight testing tool with as simple a user interface as
possible.

**lit** should be run with one or more *tests* to run specified on the command
line. Tests can be either individual test files or directories to search for
tests (see "TEST DISCOVERY").

Each specified test will be executed (potentially in parallel) and once all
tests have been run **lit** will print summary information on the number of tests
which passed or failed (see "TEST STATUS RESULTS"). The **lit** program will
execute with a non-zero exit code if any tests fail.

By default **lit** will use a succinct progress display and will only print
summary information for test failures. See "OUTPUT OPTIONS" for options
controlling the **lit** progress display and output.

**lit** also includes a number of options for controlling how tests are executed
(specific features may depend on the particular test format). See "EXECUTION
OPTIONS" for more information.

Finally, **lit** also supports additional options for only running a subset of
the options specified on the command line, see "SELECTION OPTIONS" for
more information.

Users interested in the **lit** architecture or designing a **lit** testing
implementation should see "LIT INFRASTRUCTURE"


GENERAL OPTIONS
---------------



**-h**, **--help**

 Show the **lit** help message.



**-j** *N*, **--threads**\ =\ *N*

 Run *N* tests in parallel. By default, this is automatically chosen to match
 the number of detected available CPUs.



**--config-prefix**\ =\ *NAME*

 Search for *NAME.cfg* and *NAME.site.cfg* when searching for test suites,
 instead of *lit.cfg* and *lit.site.cfg*.



**--param** *NAME*, **--param** *NAME*\ =\ *VALUE*

 Add a user defined parameter *NAME* with the given *VALUE* (or the empty
 string if not given). The meaning and use of these parameters is test suite
 dependent.




OUTPUT OPTIONS
--------------



**-q**, **--quiet**

 Suppress any output except for test failures.



**-s**, **--succinct**

 Show less output, for example don't show information on tests that pass.



**-v**, **--verbose**

 Show more information on test failures, for example the entire test output
 instead of just the test result.



**--no-progress-bar**

 Do not use curses based progress bar.




EXECUTION OPTIONS
-----------------



**--path**\ =\ *PATH*

 Specify an addition *PATH* to use when searching for executables in tests.



**--vg**

 Run individual tests under valgrind (using the memcheck tool). The
 *--error-exitcode* argument for valgrind is used so that valgrind failures will
 cause the program to exit with a non-zero status.

 When this option is enabled, **lit** will also automatically provide a
 "valgrind" feature that can be used to conditionally disable (or expect failure
 in) certain tests.



**--vg-arg**\ =\ *ARG*

 When *--vg* is used, specify an additional argument to pass to valgrind itself.



**--vg-leak**

 When *--vg* is used, enable memory leak checks. When this option is enabled,
 **lit** will also automatically provide a "vg_leak" feature that can be
 used to conditionally disable (or expect failure in) certain tests.




**--time-tests**

 Track the wall time individual tests take to execute and includes the results in
 the summary output. This is useful for determining which tests in a test suite
 take the most time to execute. Note that this option is most useful with *-j
 1*.




SELECTION OPTIONS
-----------------



**--max-tests**\ =\ *N*

 Run at most *N* tests and then terminate.



**--max-time**\ =\ *N*

 Spend at most *N* seconds (approximately) running tests and then terminate.



**--shuffle**

 Run the tests in a random order.




ADDITIONAL OPTIONS
------------------



**--debug**

 Run **lit** in debug mode, for debugging configuration issues and **lit** itself.



**--show-suites**

 List the discovered test suites as part of the standard output.



**--no-tcl-as-sh**

 Run Tcl scripts internally (instead of converting to shell scripts).



**--repeat**\ =\ *N*

 Run each test *N* times. Currently this is primarily useful for timing tests,
 other results are not collated in any reasonable fashion.




EXIT STATUS
-----------


**lit** will exit with an exit code of 1 if there are any FAIL or XPASS
results. Otherwise, it will exit with the status 0. Other exit codes are used
for non-test related failures (for example a user error or an internal program
error).


TEST DISCOVERY
--------------


The inputs passed to **lit** can be either individual tests, or entire
directories or hierarchies of tests to run. When **lit** starts up, the first
thing it does is convert the inputs into a complete list of tests to run as part
of *test discovery*.

In the **lit** model, every test must exist inside some *test suite*. **lit**
resolves the inputs specified on the command line to test suites by searching
upwards from the input path until it finds a *lit.cfg* or *lit.site.cfg*
file. These files serve as both a marker of test suites and as configuration
files which **lit** loads in order to understand how to find and run the tests
inside the test suite.

Once **lit** has mapped the inputs into test suites it traverses the list of
inputs adding tests for individual files and recursively searching for tests in
directories.

This behavior makes it easy to specify a subset of tests to run, while still
allowing the test suite configuration to control exactly how tests are
interpreted. In addition, **lit** always identifies tests by the test suite they
are in, and their relative path inside the test suite. For appropriately
configured projects, this allows **lit** to provide convenient and flexible
support for out-of-tree builds.


TEST STATUS RESULTS
-------------------


Each test ultimately produces one of the following six results:


**PASS**

 The test succeeded.



**XFAIL**

 The test failed, but that is expected. This is used for test formats which allow
 specifying that a test does not currently work, but wish to leave it in the test
 suite.



**XPASS**

 The test succeeded, but it was expected to fail. This is used for tests which
 were specified as expected to fail, but are now succeeding (generally because
 the feature they test was broken and has been fixed).



**FAIL**

 The test failed.



**UNRESOLVED**

 The test result could not be determined. For example, this occurs when the test
 could not be run, the test itself is invalid, or the test was interrupted.



**UNSUPPORTED**

 The test is not supported in this environment. This is used by test formats
 which can report unsupported tests.



Depending on the test format tests may produce additional information about
their status (generally only for failures). See the Output|"OUTPUT OPTIONS"
section for more information.


LIT INFRASTRUCTURE
------------------


This section describes the **lit** testing architecture for users interested in
creating a new **lit** testing implementation, or extending an existing one.

**lit** proper is primarily an infrastructure for discovering and running
arbitrary tests, and to expose a single convenient interface to these
tests. **lit** itself doesn't know how to run tests, rather this logic is
defined by *test suites*.

TEST SUITES
~~~~~~~~~~~


As described in "TEST DISCOVERY", tests are always located inside a *test
suite*. Test suites serve to define the format of the tests they contain, the
logic for finding those tests, and any additional information to run the tests.

**lit** identifies test suites as directories containing *lit.cfg* or
*lit.site.cfg* files (see also **--config-prefix**). Test suites are initially
discovered by recursively searching up the directory hierarchy for all the input
files passed on the command line. You can use **--show-suites** to display the
discovered test suites at startup.

Once a test suite is discovered, its config file is loaded. Config files
themselves are Python modules which will be executed. When the config file is
executed, two important global variables are predefined:


**lit**

 The global **lit** configuration object (a *LitConfig* instance), which defines
 the builtin test formats, global configuration parameters, and other helper
 routines for implementing test configurations.



**config**

 This is the config object (a *TestingConfig* instance) for the test suite,
 which the config file is expected to populate. The following variables are also
 available on the *config* object, some of which must be set by the config and
 others are optional or predefined:

 **name** *[required]* The name of the test suite, for use in reports and
 diagnostics.

 **test_format** *[required]* The test format object which will be used to
 discover and run tests in the test suite. Generally this will be a builtin test
 format available from the *lit.formats* module.

 **test_src_root** The filesystem path to the test suite root. For out-of-dir
 builds this is the directory that will be scanned for tests.

 **test_exec_root** For out-of-dir builds, the path to the test suite root inside
 the object directory. This is where tests will be run and temporary output files
 placed.

 **environment** A dictionary representing the environment to use when executing
 tests in the suite.

 **suffixes** For **lit** test formats which scan directories for tests, this
 variable is a list of suffixes to identify test files. Used by: *ShTest*,
 *TclTest*.

 **substitutions** For **lit** test formats which substitute variables into a test
 script, the list of substitutions to perform. Used by: *ShTest*, *TclTest*.

 **unsupported** Mark an unsupported directory, all tests within it will be
 reported as unsupported. Used by: *ShTest*, *TclTest*.

 **parent** The parent configuration, this is the config object for the directory
 containing the test suite, or None.

 **root** The root configuration. This is the top-most **lit** configuration in
 the project.

 **on_clone** The config is actually cloned for every subdirectory inside a test
 suite, to allow local configuration on a per-directory basis. The *on_clone*
 variable can be set to a Python function which will be called whenever a
 configuration is cloned (for a subdirectory). The function should takes three
 arguments: (1) the parent configuration, (2) the new configuration (which the
 *on_clone* function will generally modify), and (3) the test path to the new
 directory being scanned.




TEST DISCOVERY
~~~~~~~~~~~~~~


Once test suites are located, **lit** recursively traverses the source directory
(following *test_src_root*) looking for tests. When **lit** enters a
sub-directory, it first checks to see if a nested test suite is defined in that
directory. If so, it loads that test suite recursively, otherwise it
instantiates a local test config for the directory (see "LOCAL CONFIGURATION
FILES").

Tests are identified by the test suite they are contained within, and the
relative path inside that suite. Note that the relative path may not refer to an
actual file on disk; some test formats (such as *GoogleTest*) define "virtual
tests" which have a path that contains both the path to the actual test file and
a subpath to identify the virtual test.


LOCAL CONFIGURATION FILES
~~~~~~~~~~~~~~~~~~~~~~~~~


When **lit** loads a subdirectory in a test suite, it instantiates a local test
configuration by cloning the configuration for the parent direction -- the root
of this configuration chain will always be a test suite. Once the test
configuration is cloned **lit** checks for a *lit.local.cfg* file in the
subdirectory. If present, this file will be loaded and can be used to specialize
the configuration for each individual directory. This facility can be used to
define subdirectories of optional tests, or to change other configuration
parameters -- for example, to change the test format, or the suffixes which
identify test files.


TEST RUN OUTPUT FORMAT
~~~~~~~~~~~~~~~~~~~~~~


The b<lit> output for a test run conforms to the following schema, in both short
and verbose modes (although in short mode no PASS lines will be shown). This
schema has been chosen to be relatively easy to reliably parse by a machine (for
example in buildbot log scraping), and for other tools to generate.

Each test result is expected to appear on a line that matches:

<result code>: <test name> (<progress info>)

where <result-code> is a standard test result such as PASS, FAIL, XFAIL, XPASS,
UNRESOLVED, or UNSUPPORTED. The performance result codes of IMPROVED and
REGRESSED are also allowed.

The <test name> field can consist of an arbitrary string containing no newline.

The <progress info> field can be used to report progress information such as
(1/300) or can be empty, but even when empty the parentheses are required.

Each test result may include additional (multiline) log information in the
following format.

<log delineator> TEST '(<test name>)' <trailing delineator>
... log message ...
<log delineator>

where <test name> should be the name of a preceding reported test, <log
delineator> is a string of '\*' characters *at least* four characters long (the
recommended length is 20), and <trailing delineator> is an arbitrary (unparsed)
string.

The following is an example of a test run output which consists of four tests A,
B, C, and D, and a log message for the failing test C::

  PASS: A (1 of 4)
  PASS: B (2 of 4)
  FAIL: C (3 of 4)
  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* TEST 'C' FAILED \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
  Test 'C' failed as a result of exit code 1.
  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
  PASS: D (4 of 4)


LIT EXAMPLE TESTS
~~~~~~~~~~~~~~~~~


The **lit** distribution contains several example implementations of test suites
in the *ExampleTests* directory.


SEE ALSO
--------


valgrind(1)
