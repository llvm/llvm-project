## This test suite tests pipelining in-process builtins.
## This test suite has some extra in-process builtin registered in the config:
## - uppercaser: Reads input on stdin, converts the string to uppercase and
##   outputs the result on stderr.
## - print_out_err: Writes the first argument to stdout, and the second to stderr.
## - streq: Returns 0 if the first argument and the input from stdin are equal, 1 otherwise.

# RUN: %{lit} -v %{inputs}/shtest-pipelined-inproc-builtins | FileCheck %s -match-full-lines
# END.

# CHECK: PASS: shtest-pipelined-inproc-builtins :: extra-inproc-builtins.txt ({{.*}})
# CHECK: PASS: shtest-pipelined-inproc-builtins :: pipelined-inproc-builtins.txt ({{.*}})
