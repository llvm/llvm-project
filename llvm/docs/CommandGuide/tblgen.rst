tblgen - Target Description To C++ Code Generator
=================================================


SYNOPSIS
--------


**tblgen** [*options*] [*filename*]


DESCRIPTION
-----------


**tblgen** translates from target description (.td) files into C++ code that can
be included in the definition of an LLVM target library. Most users of LLVM will
not need to use this program. It is only for assisting with writing an LLVM
target backend.

The input and output of **tblgen** is beyond the scope of this short
introduction. Please see the *CodeGeneration* page in the LLVM documentation.

The *filename* argument specifies the name of a Target Description (.td) file
to read as input.


OPTIONS
-------



**-help**

 Print a summary of command line options.



**-o** *filename*

 Specify the output file name.  If *filename* is ``-``, then **tblgen**
 sends its output to standard output.



**-I** *directory*

 Specify where to find other target description files for inclusion. The
 *directory* value should be a full or partial path to a directory that contains
 target description files.



**-asmparsernum** *N*

 Make -gen-asm-parser emit assembly writer number *N*.



**-asmwriternum** *N*

 Make -gen-asm-writer emit assembly writer number *N*.



**-class** *class Name*

 Print the enumeration list for this class.



**-print-records**

 Print all records to standard output (default).



**-print-enums**

 Print enumeration values for a class



**-print-sets**

 Print expanded sets for testing DAG exprs.



**-gen-emitter**

 Generate machine code emitter.



**-gen-register-info**

 Generate registers and register classes info.



**-gen-instr-info**

 Generate instruction descriptions.



**-gen-asm-writer**

 Generate the assembly writer.



**-gen-disassembler**

 Generate disassembler.



**-gen-pseudo-lowering**

 Generate pseudo instruction lowering.



**-gen-dag-isel**

 Generate a DAG (Directed Acycle Graph) instruction selector.



**-gen-asm-matcher**

 Generate assembly instruction matcher.



**-gen-dfa-packetizer**

 Generate DFA Packetizer for VLIW targets.



**-gen-fast-isel**

 Generate a "fast" instruction selector.



**-gen-subtarget**

 Generate subtarget enumerations.



**-gen-intrinsic**

 Generate intrinsic information.



**-gen-tgt-intrinsic**

 Generate target intrinsic information.



**-gen-enhanced-disassembly-info**

 Generate enhanced disassembly info.



**-version**

 Show the version number of this program.




EXIT STATUS
-----------


If **tblgen** succeeds, it will exit with 0.  Otherwise, if an error
occurs, it will exit with a non-zero value.
