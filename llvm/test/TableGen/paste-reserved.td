// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s

defvar list1 = ["foo", "bar", "snork"];

// Pasting a list with a reserved word should produce an error.

#ifdef ERROR1
def list_paste {
  list<string> the_list = list1 # in;
}
// ERROR1: error: Unknown or reserved token when parsing a value
#endif


// Pasting an identifier with a reserved word should produce an error.

#ifdef ERROR2
def name_paste#in {
}
// ERROR2: error: Unknown or reserved token when parsing a value
#endif
