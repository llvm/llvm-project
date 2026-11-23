* RUN: llvm-mc <%s --triple s390x-ibm-zos --filetype=obj -o - | \
* RUN:   od -An -tx1 -v | tr -d ' \n' | FileCheck --ignore-case %s
* REQUIRES: systemz-registered-target
*
* HLASM continuation rules: a non-blank in column 72 continues the statement
* on the next line, which resumes in column 16.  The column 72 character is
* not part of the statement, and a keyword or a name may be split anywhere,
* including inside a PART(...) operand.
*
* "zos_cont#S" is the PART name, split across a continuation line; it must be
* rebuilt without the column 72 character (EBCDIC "zos_cont#S").
* The ESD name spans two records, so match the part before the split and the tail.
* CHECK: a996a26d839695a3{{[0-9a-f]*}}7be2
zos_cont#C CSECT
C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
C_WSA64 CATTR ALIGN(4),FILL(0),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(zosX ccc
               _cont#S)
zos_cont#S XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
 END
