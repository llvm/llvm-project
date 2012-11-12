llvm-ranlib - Generate index for LLVM archive
=============================================


SYNOPSIS
--------


**llvm-ranlib** [--version] [-help] <archive-file>


DESCRIPTION
-----------


The **llvm-ranlib** command is similar to the common Unix utility, ``ranlib``. It
adds or updates the symbol table in an LLVM archive file. Note that using the
**llvm-ar** modifier *s* is usually more efficient than running **llvm-ranlib**
which is only provided only for completness and compatibility. Unlike other
implementations of ``ranlib``, **llvm-ranlib** indexes LLVM bitcode files, not
native object modules. You can list the contents of the symbol table with the
``llvm-nm -s`` command.


OPTIONS
-------



*archive-file*

 Specifies the archive-file to which the symbol table is added or updated.



*--version*

 Print the version of **llvm-ranlib** and exit without building a symbol table.



*-help*

 Print usage help for **llvm-ranlib** and exit without building a symbol table.




EXIT STATUS
-----------


If **llvm-ranlib** succeeds, it will exit with 0.  If an error occurs, a non-zero
exit code will be returned.


SEE ALSO
--------


llvm-ar|llvm-ar, ranlib(1)
