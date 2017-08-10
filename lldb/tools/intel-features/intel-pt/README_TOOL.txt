*******************************************************************************
*                            README                                           *
*                                                                             *
* This file provides all the information regarding Intel(R) Processor Trace   *
* Tool. It consists explanation about how Tool internally works, its hardware *
* and software dependencies, build procedure and usage of the API.            *
*******************************************************************************



============
Introduction
============
The Intel(R) Processor Trace Tool is developed on top of LLDB and provides its
its users execution trace of the debugged applications. Tool makes use of
Intel(R) Processor Trace hardware feature implementation inside LLDB for this
purpose. This hardware feature generates a set of trace packets that
encapsulates program flow information. These trace packets along with the binary
of the application can be decoded with the help of a software decoder to
construct the execution trace of the application.

More information about Intel(R) Processor Trace feature can be obtained from
website: https://software.intel.com/en-us/blogs/2013/09/18/processor-tracing




=========
Details
=========
The functionality of the Tool consists three parts:

1. Raw Trace Collection from LLDB
      With the help of API of this Tool (given below), Intel(R) Processor Trace
      can be started on the application being debugged with LLDB. The generated
      trace of the application is gathered inside LLDB and is collected by the
      Tool from LLDB through LLDB's public API.

2. Raw Trace Decoding
      For decoding the raw trace data, the Tool makes use of "libipt", an
      Intel(R) Processor Trace Decoder Library. The library needs binary of
      the application and information about the cpu on which the application is
      running in order to decode the raw trace. The Tool gathers this
      information from LLDB public API and provide it to "libipt". More
      information about "libipt" can be found at:
      https://software.intel.com/en-us/blogs/2013/09/18/processor-tracing and
      https://github.com/01org/processor-trace

3. Decoded Trace Post-processing
      The decoded trace is post-processed to reconstruct the execution flow of
      the application. The execution flow contains the list of assembly
      instructions (called instruction log hereafter).




=============
Dependencies
=============
The Tool has following hardware and software dependencies:

  - Hardware dependency: The Tool makes use of this hardware feature to capture
    raw trace of an application from LLDB. This hardware feature may not be
    present in all processors. The hardware feature is supported on Broadwell
    and other succeeding CPUs such as Skylake etc. In order for Tool to provide
    something meaningful, the target machine on which the application is running
    should have this feature.

  - Software dependency: The Tool has an indirect dependency on the Operating
    System level software support for Intel(R) Processor Trace on the target
    machine where the application is running and being debugged by LLDB. This
    support is required to enable raw trace generation on the target machine.
    Currently, the Tool works for applications running on Linux OS as till now
    the Operating System level support for the feature is present only in Linux
    (more specifically starting from the 4.1 kernel). In Linux, this feature is
    implemented in perf_events subsystem and is usable through perf_event_open
    system call. In the User space level, the Tool has a direct dependency on
    "libipt" to decode the captured raw trace. This library might be
    pre-installed on host systems. If not then the library can be built from
    its sources (available at): https://github.com/01org/processor-trace




============
How to Build
============
The Tool has a cmake based build and can be built by specifying some extra flags
while building LLDB with cmake. The following cmake flags need to be provided to
build the Tool:

  - LIBIPT_INCLUDE_PATH - The flag specifies the directory where the header
    file of "libipt" resides. If the library is not pre-installed on the host
    system and is built directly from "libipt" project sources then this file
    may either come as a part of the sources itself or will be generated in
    build folder while building library.

  - LIBIPT_LIBRARY_PATH - The flag points to the location of "libipt" shared
    library.

The Tool currently works successfully with following versions of this library:
  - v1.4, v1.5, v1.6



============
How to Use
============
The Tool's API are exposed as a C++ object oriented interface (file PTDecoder.h)
in a shared library. The main class that implements the whole functionality is
PTDecoder. This class makes use of 3 other classes,
 - PTInstruction to represent an assembly instruction
 - PTInstructionList to return instruction log
 - PTTraceOptions to return trace specific information
The users can use these API to develop their own products. All API are also
available as python functions through a script bridging interface, allowing
them to be used directly from python either interactively or to build python
apps.

Currently, cli wrapper has been developed on top of the Tool to use it through
LLDB's command line. Please refer to README_CLI.txt file for command line usage.


A brief introduction about the classes and their API are given below.

  class PTDecoder
  ===============
    This class makes use of Intel(R) Processor Trace hardware feature
    (implemented inside LLDB) to gather trace data for an inferior (being
    debugged with LLDB) to provide meaningful information out of it. Currently
    the meaningful information comprises of the execution flow of the inferior
    (in terms of assembly instructions executed). The class enables user to:

    - start the trace with configuration options for a thread/process,
    - stop the trace for a thread/process,
    - get the execution flow (assembly instructions) for a thread and
    - get trace specific information for a thread

    Corresponding API are explained below:
    a) void StartProcessorTrace(lldb::SBProcess &sbprocess,
                                lldb::SBTraceOptions &sbtraceoptions,
                                lldb::SBError &sberror)
       ------------------------------------------------------------------------
           This API allows the user to start trace on a particular thread or on
           the whole process with Intel(R) Processor Trace specific
           configuration options.

           @param[in] sbprocess      : A valid process on which this operation
               will be performed. An error is returned in case of an invalid
               process.

           @param[out] sberror       : An error with the failure reason if API
               fails. Else success.

           @param[in] sbtraceoptions : Contains thread id information and
               configuration options:
               For tracing a single thread, provide a valid thread id. If
               sbprocess doesn't contain this thread id, error will be returned.
               For tracing complete process, set to lldb::LLDB_INVALID_THREAD_ID
               Configuration options comprises of:
                - trace buffer size, meta data buffer size, TraceType and
                - All other possible Intel(R) Processor Trace specific
                  configuration options (hereafter collectively referred as
                  CUSTOM_OPTIONS)

                Trace buffer, meant to store the trace data read from target
                machine, inside LLDB is configured as a cyclic buffer. Hence,
                depending upon the trace buffer size provided here, buffer
                overwrites may happen while LLDB writes trace data into it.
                CUSTOM_OPTIONS are formatted as json text i.e. {"Name":Value,
                "Name":Value,...} inside sbtraceoptions, where "Value" should be
                a 64-bit unsigned integer in hex format. For information
                regarding what all configuration options are currently supported
                by LLDB and detailed information about CUSTOM_OPTIONS usage,
                please refer to SBProcess::StartTrace() API description. An
                overview of some of the various CUSTOM_OPTIONS are briefly given
                below. Please refer to "Intel(R) 64 and IA-32 Architectures
                Software Developer's Manual" for more details about them.
                  - CYCEn       Enable/Disable Cycle Count Packet (CYC) Packet
                  - OS          Packet generation enabled/disabled if
                                Current Privilege Level (CPL)=0
                  - User        Packet generation enabled/disabled if CPL>0
                  - CR3Filter   Enable/Disable CR3 Filtering
                  - MTCEn       Enable/disable MTC packets
                  - TSCEn       Enable/disable TSC packets
                  - DisRETC     Enable/disable RET Compression
                  - BranchEn    Enable/disable COFI-based packets
                  - MTCFreq     Defines MTC Packet Frequency
                  - CycThresh   CYC Packet threshold
                  - PSBFreq     Frequency of PSB Packets

                TraceType should be set to
                lldb::TraceType::eTraceTypeProcessorTrace, else error is
                returned. To find out any other requirement to start tracing
                successfully, refer to SBProcess::StartTrace() API description.
                LLDB's current implementation of Intel(R) Processor Trace
                feature may round off invalid values for configuration options.
                Therefore, the configuration options with which the trace was
                actually started, might be different to the ones with which
                trace was asked to be started by user. The actual used
                configuration options can be obtained from
                GetProcessorTraceInfo() API.



    b) void StopProcessorTrace(lldb::SBProcess &sbprocess,
                               lldb::SBError &sberror,
                               lldb::tid_t tid = LLDB_INVALID_THREAD_ID)
       ------------------------------------------------------------------------
           This API allows the user to Stop trace on a particular thread or on
           the whole process.

           @param[in] sbprocess : A valid process on which this operation will
               be performed. An error is returned in case of an invalid process.

           @param[in] tid       : To stop tracing a single thread, provide a
               valid thread id. If sbprocess doesn't contain the thread tid,
               error will be returned. To stop tracing complete process, use
               lldb::LLDB_INVALID_THREAD_ID

           @param[out] sberror  : An error with the failure reason if API fails.
               Else success



    c) void GetInstructionLogAtOffset(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                                      uint32_t offset, uint32_t count,
                                      PTInstructionList &result_list,
                                      lldb::SBError &sberror)
       ------------------------------------------------------------------------
           This API provides instruction log that contains the execution flow
           for a thread of a process in terms of assembly instruction executed.
           The API works on only 1 thread at a time. To gather this information
           for whole process, this API needs to be called for each thread.

           @param[in] sbprocess    : A valid process on which this operation
               will be performed. An error is returned in case of an invalid
               process.

           @param[in] tid          : A valid thread id of the thread for which
               instruction log is desired. If sbprocess doesn't contain the
               thread tid, error will be returned.

           @param[in] count        : Number of instructions requested by the
               user to be returned from the complete instruction log. Complete
               instruction log refers to all the assembly instructions obtained
               after decoding the complete raw trace data obtained from LLDB.
               The length of the complete instruction log is dependent on the
               trace buffer size with which processor tracing was started for
               this thread.
               The number of instructions actually returned are dependent on
               'count' and 'offset' parameters of this API.

           @param[in] offset       : The offset in the complete instruction log
               from where 'count' number of instructions are requested by the
               user. offset is counted from the end of of this complete
               instruction log (which means the last executed instruction
               is at offset 0 (zero)).

           @param[out] result_list : Depending upon 'count' and 'offset' values,
               list will be overwritten with the instructions.

           @param[out] sberror     : An error with the failure reason if API
               fails. Else success



    d) void GetProcessorTraceInfo(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                                  PTTraceOptions &options, lldb::SBError &sberror)
       ------------------------------------------------------------------------
           This API provides Intel(R) Processor Trace specific information for
           a thread of a process. The API works on only 1 thread at a time. To
           gather this information for whole process, this API needs to be
           called for each thread. The information contains the actual
           configuration options with which the trace was started for this
           thread.

           @param[in] sbprocess  : The valid process on which this operation
               will be performed. An error is returned in case of an invalid
               process.

           @param[in] tid        : A valid thread id of the thread for which the
               trace specific information is required. If sbprocess doesn't
               contain the thread tid, an error will be returned.

           @param[out] options   : Contains actual configuration options (they
               may be different to the ones with which tracing was asked to be
               started for this thread during StartProcessorTrace() API call).

           @param[out] sberror   : An error with the failure reason if API
               fails. Else success


  class PTInstruction
  ===================
      This class represents an assembly instruction containing raw instruction
      bytes, instruction address along with execution flow context and
      Intel(R) Processor Trace context. For more details, please refer to
      PTDecoder.h file.

  class PTInstructionList
  =======================
      This class represents a list of assembly instructions. Each assembly
      instruction is of type PTInstruction.

  class PTTraceOptions
  ====================
      This class provides Intel(R) Processor Trace specific configuration
      options like trace type, trace buffer size, meta data buffer size along
      with other trace specific options. For more details, please refer to
      PTDecoder.h file.
