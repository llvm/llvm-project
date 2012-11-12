llvm-prof - print execution profile of LLVM program
===================================================


SYNOPSIS
--------


**llvm-prof** [*options*] [*bitcode file*] [*llvmprof.out*]


DESCRIPTION
-----------


The **llvm-prof** tool reads in an *llvmprof.out* file (which can
optionally use a specific file with the third program argument), a bitcode file
for the program, and produces a human readable report, suitable for determining
where the program hotspots are.

This program is often used in conjunction with the *utils/profile.pl*
script.  This script automatically instruments a program, runs it with the JIT,
then runs **llvm-prof** to format a report.  To get more information about
*utils/profile.pl*, execute it with the **-help** option.


OPTIONS
-------



**--annotated-llvm** or **-A**

 In addition to the normal report printed, print out the code for the
 program, annotated with execution frequency information. This can be
 particularly useful when trying to visualize how frequently basic blocks
 are executed.  This is most useful with basic block profiling
 information or better.



**--print-all-code**

 Using this option enables the **--annotated-llvm** option, but it
 prints the entire module, instead of just the most commonly executed
 functions.



**--time-passes**

 Record the amount of time needed for each pass and print it to standard
 error.




EXIT STATUS
-----------


**llvm-prof** returns 1 if it cannot load the bitcode file or the profile
information. Otherwise, it exits with zero.
