****************************************************************************
*                            README                                        *
*                                                                          *
* This file provides all the information regarding 4 new CLI commands that *
* enable using Intel(R) Processor Trace Tool from LLDB's CLI.              *
****************************************************************************


============
Introduction
============
A C++ based cli wrapper has been developed to use Intel(R) Processor Trace Tool
through LLDB's command line. This also provides an idea to all developers on how
to integrate the Tool into various IDEs providing LLDB as a debugger.



============
How to Build
============
The wrapper cli-wrapper-pt.cpp needs to be compiled and linked with the shared
library of the Intel(R) Processor Trace Tool in order to be used through LLDB's
CLI. The procedure to build shared library of the Intel(R) Processor Trace Tool
is given in README_TOOL.txt file.



============
How to Use
============
All these commands are available via shared library (lldbIntelFeatures)
obtained after building intel-features folder from top. Please refer to
cli-wrapper.cpp and README files of "intel-features" folder for this purpose.



============
Description
============
4 CLI commands have been designed keeping the LLDB's existing CLI command syntax
in mind.

   1) processor-trace start [-b <buffer-size>] [<thread-index>]

      Start Intel(R) Processor Trace on a specific thread or on the whole process

      Syntax: processor-trace start  <cmd-options>

      cmd-options Usage:
        processor-trace start [-b <buffer-size>] [<thread-index>]

          -b <buffer-size>
             size of the trace buffer to store the trace data. If not specified
             then a default value (=4KB) will be taken

          <thread-index>
             thread index of the thread. If no threads are specified, currently
             selected thread is taken. Use the thread-index 'all' to start
             tracing the whole process



   2) processor-trace stop [<thread-index>]

      Stop Intel(R) Processor Trace on a specific thread or on the whole process

      Syntax: processor-trace stop  <cmd-options>

      cmd-options Usage:
      processor-trace stop [<thread-index>]

          <thread-index>
             thread index of the thread. If no threads are specified, currently
             selected thread is taken. Use the thread-index 'all' to stop
             tracing the whole process



   3) processor-trace show-trace-options [<thread-index>]

      Display all the information regarding Intel(R) Processor Trace for a specific
      thread or for the whole process. The information contains trace buffer
      size and configuration options of Intel(R) Processor Trace.

      Syntax: processor-trace show-trace-options <cmd-options>

      cmd-options Usage:
        processor-trace show-trace-options [<thread-index>]

          <thread-index>
             thread index of the thread. If no threads are specified, currently
             selected thread is taken. Use the thread-index 'all' to display
             information for all threads of the process



   4) processor-trace show-instr-log [-o <offset>] [-c <count>] [<thread-index>]

      Display a log of assembly instructions executed for a specific thread or
      for the whole process. The length of the log to be displayed and the
      offset in the whole instruction log from where the log needs to be
      displayed can also be provided. The offset is counted from the end of this
      whole instruction log which means the last executed instruction is at
      offset 0 (zero).

      Syntax: processor-trace show-instr-log  <cmd-options>

      cmd-options Usage:
        processor-trace show-instr-log [-o <offset>] [-c <count>] [<thread-index>]

          -c <count>
             number of instructions to be displayed. If not specified then a
             default value (=10) will be taken

          -o <offset>
             offset in the whole instruction log from where the log will be
             displayed. If not specified then default value is calculated as
             offset = count -1

          <thread-index>
             thread index of the thread. If no threads are specified, currently
             selected thread is taken. Use the thread-index 'all' to show
             instruction log for all the threads of the process
