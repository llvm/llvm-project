int foo() { return 123; }

#ifndef ONLY_CLANG
syntax_error_for_lldb_to_find // comment that tests source printing
#endif
