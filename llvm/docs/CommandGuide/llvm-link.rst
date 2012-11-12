llvm-link - LLVM linker
=======================


SYNOPSIS
--------


**llvm-link** [*options*] *filename ...*


DESCRIPTION
-----------


**llvm-link** takes several LLVM bitcode files and links them together into a
single LLVM bitcode file.  It writes the output file to standard output, unless
the **-o** option is used to specify a filename.

**llvm-link** attempts to load the input files from the current directory.  If
that fails, it looks for each file in each of the directories specified by the
**-L** options on the command line.  The library search paths are global; each
one is searched for every input file if necessary.  The directories are searched
in the order they were specified on the command line.


OPTIONS
-------



**-L** *directory*

 Add the specified *directory* to the library search path.  When looking for
 libraries, **llvm-link** will look in path name for libraries.  This option can be
 specified multiple times; **llvm-link** will search inside these directories in
 the order in which they were specified on the command line.



**-f**

 Enable binary output on terminals.  Normally, **llvm-link** will refuse to
 write raw bitcode output if the output stream is a terminal. With this option,
 **llvm-link** will write raw bitcode regardless of the output device.



**-o** *filename*

 Specify the output file name.  If *filename* is ``-``, then **llvm-link** will
 write its output to standard output.



**-S**

 Write output in LLVM intermediate language (instead of bitcode).



**-d**

 If specified, **llvm-link** prints a human-readable version of the output
 bitcode file to standard error.



**-help**

 Print a summary of command line options.



**-v**

 Verbose mode.  Print information about what **llvm-link** is doing.  This
 typically includes a message for each bitcode file linked in and for each
 library found.




EXIT STATUS
-----------


If **llvm-link** succeeds, it will exit with 0.  Otherwise, if an error
occurs, it will exit with a non-zero value.


SEE ALSO
--------


gccld|gccld
