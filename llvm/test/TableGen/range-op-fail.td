// Each RUN line is scattered.

defvar list_int = !range(4);

#ifdef ERR_LIST_INT
// RUN: not llvm-tblgen %s -DERR_LIST_INT 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR_LIST_INT
// ERR_LIST_INT: [[FILE]]:[[@LINE+1]]:32: error: expected one list, got extra value of type 'int'
defvar errs = !range(list_int, 42);
#endif

#ifdef ERR_INT_LIST
// RUN: not llvm-tblgen %s -DERR_INT_LIST 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR_INT_LIST
// ERR_INT_LIST: [[FILE]]:[[@LINE+1]]:25: error: expected value of type 'int', got 'list<int>'
defvar errs = !range(0, list_int);
#endif

#ifdef ERR_TOO_MANY_ARGS
// RUN: not llvm-tblgen %s -DERR_TOO_MANY_ARGS 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR_TOO_MANY_ARGS
// ERR_TOO_MANY_ARGS: [[FILE]]:[[@LINE+1]]:34: error: expected at most three values of integer
defvar errs = !range(0, 42, 255, 233);
#endif

#ifdef ERR_UNEXPECTED_TYPE_0
// RUN: not llvm-tblgen %s -DERR_UNEXPECTED_TYPE_0 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR_UNEXPECTED_TYPE_0
// ERR_UNEXPECTED_TYPE_0: [[FILE]]:[[@LINE+1]]:22: error: expected list or int, got value of type 'string'
defvar errs = !range("hoge");
#endif

#ifdef ERR_UNEXPECTED_TYPE_1
// RUN: not llvm-tblgen %s -DERR_UNEXPECTED_TYPE_1 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR_UNEXPECTED_TYPE_1
// ERR_UNEXPECTED_TYPE_1: [[FILE]]:[[@LINE+1]]:22: error: expected list or int, got value of type 'string'
defvar errs = !range("hoge", 42);
#endif

#ifdef ERR_UNEXPECTED_TYPE_2
// RUN: not llvm-tblgen %s -DERR_UNEXPECTED_TYPE_2 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR_UNEXPECTED_TYPE_2
// ERR_UNEXPECTED_TYPE_2: [[FILE]]:[[@LINE+1]]:25: error: expected value of type 'int', got 'string'
defvar errs = !range(6, "fuga");
#endif

#ifdef ERR_EMPTY_ARG
// RUN: not llvm-tblgen %s -DERR_EMPTY_ARG 2>&1 | FileCheck -DFILE=%s %s --check-prefix=ERR_EMPTY_ARG
// ERR_EMPTY_ARG: [[FILE]]:[[@LINE+1]]:22: error: Unknown or reserved token when parsing a value
defvar errs = !range();
#endif
