// Each RUN line is scattered.

defvar list_int = [0, 1, 2, 3, 4, 5];
defvar list_str = ["foo", "bar"];
defvar str = "hoge";
defvar und = ?;

#ifdef ERR0
// RUN: not llvm-tblgen %s -DERR0 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR0
// ERR0: [[FILE]]:[[@LINE+1]]:26: error: Unknown or reserved token when parsing a value
defvar errs = list_str [ ] ;
#endif

#ifdef ERR1
// RUN: not llvm-tblgen %s -DERR1 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR1
// ERR1: [[FILE]]:[[@LINE+1]]:26: error: Unknown or reserved token when parsing a value
defvar errs = list_str [ , ] ;
#endif

#ifdef ERR2
// RUN: not llvm-tblgen %s -DERR2 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR2
// ERR2: [[FILE]]:[[@LINE+1]]:26: error: expected list<int>, got list<string>
defvar errs = list_str [ list_str ] ;
#endif

#ifdef ERR3
// RUN: not llvm-tblgen %s -DERR3 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR3
// ERR3: [[FILE]]:[[@LINE+1]]:26: error: expected int...int, got list<string>
defvar errs = list_str [ list_str ... 42 ] ;
#endif

#ifdef ERR4
// RUN: not llvm-tblgen %s -DERR4 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR4
// ERR4: [[FILE]]:[[@LINE+1]]:32: error: expected int...int, got list<string>
defvar errs = list_str [ 0 ... list_str ] ;
#endif

#ifdef ERR5
// RUN: not llvm-tblgen %s -DERR5 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR5
// ERR5: [[FILE]]:[[@LINE+1]]:26: error: unhandled type string in range
defvar errs = list_str [ str ] ;
#endif

#ifdef ERR6
// RUN: not llvm-tblgen %s -DERR6 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR6
// ERR6: [[FILE]]:[[@LINE+1]]:28: error: invalid range, cannot be negative
defvar errs = list_str [ 5 1 ] ;
#endif

#ifdef ERR7
// RUN: not llvm-tblgen %s -DERR7 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR7
// ERR7: [[FILE]]:[[@LINE+1]]:15: error: Type 'string' is invalid, list expected
defvar errs = str [ 0 ] ;
#endif

#ifdef ERR8
// RUN: not llvm-tblgen %s -DERR8 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR8
// ERR8: [[FILE]]:[[@LINE+1]]:26: error: Unknown or reserved token when parsing a value
defvar errs = list_int [ ... ] ;
#endif

#ifdef ERR9
// RUN: not llvm-tblgen %s -DERR9 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR9
// ERR9: [[FILE]]:[[@LINE+1]]:32: error: Unknown or reserved token when parsing a value
defvar errs = list_int [ 0 ... ] ;
#endif

#ifdef ERRA
// RUN: not llvm-tblgen %s -DERRA 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERRA
// ERRA: [[FILE]]:[[@LINE+1]]:15: error: Invalid value, list expected
defvar errs = und [ 0 ] ;
#endif
