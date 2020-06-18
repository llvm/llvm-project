# Check the internal shell handling component of the ShTest format.
#
# RUN: not %{lit} -j 1 -v %{inputs}/shtest-shell > %t.out
# FIXME: Temporarily dump test output so we can debug failing tests on
# buildbots.
# RUN: cat %t.out
# RUN: FileCheck --input-file %t.out %s
#
# END.

# CHECK: -- Testing:

# CHECK: FAIL: shtest-shell :: cat-error-0.txt
# CHECK: *** TEST 'shtest-shell :: cat-error-0.txt' FAILED ***
# CHECK: $ "cat" "-b" "temp1.txt"
# CHECK: # command stderr:
# CHECK: Unsupported: 'cat':  option -b not recognized
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: cat-error-1.txt
# CHECK: *** TEST 'shtest-shell :: cat-error-1.txt' FAILED ***
# CHECK: $ "cat" "temp1.txt"
# CHECK: # command stderr:
# CHECK: [Errno 2] No such file or directory: 'temp1.txt'
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: colon-error.txt
# CHECK: *** TEST 'shtest-shell :: colon-error.txt' FAILED ***
# CHECK: $ ":"
# CHECK: # command stderr:
# CHECK: Unsupported: ':' cannot be part of a pipeline
# CHECK: error: command failed with exit status: 127
# CHECK: ***

# CHECK: PASS: shtest-shell :: dev-null.txt

# CHECK: FAIL: shtest-shell :: diff-b.txt
# CHECK: *** TEST 'shtest-shell :: diff-b.txt' FAILED ***
# CHECK: $ "diff" "-b" "{{[^"]*}}.0" "{{[^"]*}}.1"
# CHECK: # command output:
# CHECK: 1,2
# CHECK-NEXT: {{^  }}f o o
# CHECK-NEXT: ! b a r
# CHECK-NEXT: ---
# CHECK-NEXT: {{^  }}f o o
# CHECK-NEXT: ! bar
# CHECK-EMPTY:
# CHECK: error: command failed with exit status: 1
# CHECK: ***


# CHECK: FAIL: shtest-shell :: diff-encodings.txt
# CHECK: *** TEST 'shtest-shell :: diff-encodings.txt' FAILED ***

# CHECK: $ "diff" "-u" "diff-in.bin" "diff-in.bin"
# CHECK-NOT: error

# CHECK: $ "diff" "-u" "diff-in.utf16" "diff-in.bin"
# CHECK: # command output:
# CHECK-NEXT: ---
# CHECK-NEXT: +++
# CHECK-NEXT: @@
# CHECK-NEXT: {{^ .f.o.o.$}}
# CHECK-NEXT: {{^-.b.a.r.$}}
# CHECK-NEXT: {{^\+.b.a.r..}}
# CHECK-NEXT: {{^ .b.a.z.$}}
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "diff" "-u" "diff-in.utf8" "diff-in.bin"
# CHECK: # command output:
# CHECK-NEXT: ---
# CHECK-NEXT: +++
# CHECK-NEXT: @@
# CHECK-NEXT: -foo
# CHECK-NEXT: -bar
# CHECK-NEXT: -baz
# CHECK-NEXT: {{^\+.f.o.o.$}}
# CHECK-NEXT: {{^\+.b.a.r..}}
# CHECK-NEXT: {{^\+.b.a.z.$}}
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "diff" "-u" "diff-in.bin" "diff-in.utf8"
# CHECK: # command output:
# CHECK-NEXT: ---
# CHECK-NEXT: +++
# CHECK-NEXT: @@
# CHECK-NEXT: {{^\-.f.o.o.$}}
# CHECK-NEXT: {{^\-.b.a.r..}}
# CHECK-NEXT: {{^\-.b.a.z.$}}
# CHECK-NEXT: +foo
# CHECK-NEXT: +bar
# CHECK-NEXT: +baz
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "cat" "diff-in.bin"
# CHECK-NOT: error
# CHECK: $ "diff" "-u" "-" "diff-in.bin"
# CHECK-NOT: error

# CHECK: $ "cat" "diff-in.bin"
# CHECK-NOT: error
# CHECK: $ "diff" "-u" "diff-in.bin" "-"
# CHECK-NOT: error

# CHECK: $ "cat" "diff-in.bin"
# CHECK-NOT: error
# CHECK: $ "diff" "-u" "diff-in.utf16" "-"
# CHECK: # command output:
# CHECK-NEXT: ---
# CHECK-NEXT: +++
# CHECK-NEXT: @@
# CHECK-NEXT: {{^ .f.o.o.$}}
# CHECK-NEXT: {{^-.b.a.r.$}}
# CHECK-NEXT: {{^\+.b.a.r..}}
# CHECK-NEXT: {{^ .b.a.z.$}}
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "cat" "diff-in.bin"
# CHECK-NOT: error
# CHECK: $ "diff" "-u" "diff-in.utf8" "-"
# CHECK: # command output:
# CHECK-NEXT: ---
# CHECK-NEXT: +++
# CHECK-NEXT: @@
# CHECK-NEXT: -foo
# CHECK-NEXT: -bar
# CHECK-NEXT: -baz
# CHECK-NEXT: {{^\+.f.o.o.$}}
# CHECK-NEXT: {{^\+.b.a.r..}}
# CHECK-NEXT: {{^\+.b.a.z.$}}
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "diff" "-u" "-" "diff-in.utf8"
# CHECK: # command output:
# CHECK-NEXT: ---
# CHECK-NEXT: +++
# CHECK-NEXT: @@
# CHECK-NEXT: {{^\-.f.o.o.$}}
# CHECK-NEXT: {{^\-.b.a.r..}}
# CHECK-NEXT: {{^\-.b.a.z.$}}
# CHECK-NEXT: +foo
# CHECK-NEXT: +bar
# CHECK-NEXT: +baz
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "false"

# CHECK: ***


# CHECK: FAIL: shtest-shell :: diff-error-1.txt
# CHECK: *** TEST 'shtest-shell :: diff-error-1.txt' FAILED ***
# CHECK: $ "diff" "-B" "temp1.txt" "temp2.txt"
# CHECK: # command stderr:
# CHECK: Unsupported: 'diff': option -B not recognized
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: diff-error-2.txt
# CHECK: *** TEST 'shtest-shell :: diff-error-2.txt' FAILED ***
# CHECK: $ "diff" "temp.txt"
# CHECK: # command stderr:
# CHECK: Error: missing or extra operand
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: diff-error-3.txt
# CHECK: *** TEST 'shtest-shell :: diff-error-3.txt' FAILED ***
# CHECK: $ "diff" "temp.txt" "temp1.txt"
# CHECK: # command stderr:
# CHECK: Error: 'diff' command failed
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: diff-error-4.txt
# CHECK: *** TEST 'shtest-shell :: diff-error-4.txt' FAILED ***
# CHECK: Exit Code: 1
# CHECK: # command output:
# CHECK: diff-error-4.txt.tmp
# CHECK: diff-error-4.txt.tmp1
# CHECK: *** 1 ****
# CHECK: ! hello-first
# CHECK: --- 1 ----
# CHECK: ! hello-second
# CHECK: ***

# CHECK: FAIL: shtest-shell :: diff-error-5.txt
# CHECK: *** TEST 'shtest-shell :: diff-error-5.txt' FAILED ***
# CHECK: $ "diff"
# CHECK: # command stderr:
# CHECK: Error: missing or extra operand
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: diff-error-6.txt
# CHECK: *** TEST 'shtest-shell :: diff-error-6.txt' FAILED ***
# CHECK: $ "diff"
# CHECK: # command stderr:
# CHECK: Error: missing or extra operand
# CHECK: error: command failed with exit status: 1
# CHECK: ***


# CHECK: FAIL: shtest-shell :: diff-pipes.txt

# CHECK: *** TEST 'shtest-shell :: diff-pipes.txt' FAILED ***

# CHECK: $ "diff" "{{[^"]*}}.foo" "{{[^"]*}}.foo"
# CHECK-NOT: note
# CHECK-NOT: error
# CHECK: $ "FileCheck"
# CHECK-NOT: note
# CHECK-NOT: error

# CHECK: $ "diff" "-u" "{{[^"]*}}.foo" "{{[^"]*}}.bar"
# CHECK: note: command had no output on stdout or stderr
# CHECK: error: command failed with exit status: 1
# CHECK: $ "FileCheck"
# CHECK-NOT: note
# CHECK-NOT: error
# CHECK: $ "true"

# CHECK: $ "cat" "{{[^"]*}}.foo"
# CHECK: $ "diff" "-u" "-" "{{[^"]*}}.foo"
# CHECK-NOT: note
# CHECK-NOT: error

# CHECK: $ "cat" "{{[^"]*}}.foo"
# CHECK: $ "diff" "-u" "{{[^"]*}}.foo" "-"
# CHECK-NOT: note
# CHECK-NOT: error

# CHECK: $ "cat" "{{[^"]*}}.bar"
# CHECK: $ "diff" "-u" "{{[^"]*}}.foo" "-"
# CHECK: # command output:
# CHECK: @@
# CHECK-NEXT: -foo
# CHECK-NEXT: +bar
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "cat" "{{[^"]*}}.bar"
# CHECK: $ "diff" "-u" "-" "{{[^"]*}}.foo"
# CHECK: # command output:
# CHECK: @@
# CHECK-NEXT: -bar
# CHECK-NEXT: +foo
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "cat" "{{[^"]*}}.foo"
# CHECK: $ "diff" "-" "{{[^"]*}}.foo"
# CHECK-NOT: note
# CHECK-NOT: error
# CHECK: $ "FileCheck"
# CHECK-NOT: note
# CHECK-NOT: error

# CHECK: $ "cat" "{{[^"]*}}.bar"
# CHECK: $ "diff" "-u" "{{[^"]*}}.foo" "-"
# CHECK: note: command had no output on stdout or stderr
# CHECK: error: command failed with exit status: 1
# CHECK: $ "FileCheck"
# CHECK-NOT: note
# CHECK-NOT: error
# CHECK: $ "true"

# CHECK: $ "false"

# CHECK: ***


# CHECK: FAIL: shtest-shell :: diff-r-error-0.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-0.txt' FAILED ***
# CHECK: $ "diff" "-r" 
# CHECK: # command output:
# CHECK: Only in {{.*}}dir1: dir1unique
# CHECK: Only in {{.*}}dir2: dir2unique
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-1.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-1.txt' FAILED ***
# CHECK: $ "diff" "-r" 
# CHECK: # command output:
# CHECK: *** {{.*}}dir1{{.*}}subdir{{.*}}f01
# CHECK: --- {{.*}}dir2{{.*}}subdir{{.*}}f01
# CHECK: 12345
# CHECK: 00000
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-2.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-2.txt' FAILED ***
# CHECK: $ "diff" "-r" 
# CHECK: # command output:
# CHECK: Only in {{.*}}dir2: extrafile
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-3.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-3.txt' FAILED ***
# CHECK: $ "diff" "-r" 
# CHECK: # command output:
# CHECK: Only in {{.*}}dir1: extra_subdir
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-4.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-4.txt' FAILED ***
# CHECK: $ "diff" "-r" 
# CHECK: # command output:
# CHECK: File {{.*}}dir1{{.*}}extra_subdir is a directory while file {{.*}}dir2{{.*}}extra_subdir is a regular file
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-5.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-5.txt' FAILED ***
# CHECK: $ "diff" "-r" 
# CHECK: # command output:
# CHECK: Only in {{.*}}dir1: extra_subdir
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-6.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-6.txt' FAILED ***
# CHECK: $ "diff" "-r" 
# CHECK: # command output:
# CHECK: File {{.*}}dir1{{.*}}extra_file is a regular empty file while file {{.*}}dir2{{.*}}extra_file is a directory
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-7.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-7.txt' FAILED ***
# CHECK: $ "diff" "-r" "-" "{{[^"]*}}"
# CHECK: # command stderr:
# CHECK: Error: cannot recursively compare '-'
# CHECK: error: command failed with exit status: 1

# CHECK: FAIL: shtest-shell :: diff-r-error-8.txt
# CHECK: *** TEST 'shtest-shell :: diff-r-error-8.txt' FAILED ***
# CHECK: $ "diff" "-r" "{{[^"]*}}" "-"
# CHECK: # command stderr:
# CHECK: Error: cannot recursively compare '-'
# CHECK: error: command failed with exit status: 1

# CHECK: PASS: shtest-shell :: diff-r.txt


# CHECK: FAIL: shtest-shell :: diff-strip-trailing-cr.txt

# CHECK: *** TEST 'shtest-shell :: diff-strip-trailing-cr.txt' FAILED ***

# CHECK: $ "diff" "-u" "diff-in.dos" "diff-in.unix"
# CHECK: # command output:
# CHECK: @@
# CHECK-NEXT: -In this file, the
# CHECK-NEXT: -sequence "\r\n"
# CHECK-NEXT: -terminates lines.
# CHECK-NEXT: +In this file, the
# CHECK-NEXT: +sequence "\n"
# CHECK-NEXT: +terminates lines.
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "diff" "-u" "diff-in.unix" "diff-in.dos"
# CHECK: # command output:
# CHECK: @@
# CHECK-NEXT: -In this file, the
# CHECK-NEXT: -sequence "\n"
# CHECK-NEXT: -terminates lines.
# CHECK-NEXT: +In this file, the
# CHECK-NEXT: +sequence "\r\n"
# CHECK-NEXT: +terminates lines.
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "diff" "-u" "--strip-trailing-cr" "diff-in.dos" "diff-in.unix"
# CHECK: # command output:
# CHECK: @@
# CHECK-NEXT:  In this file, the
# CHECK-NEXT: -sequence "\r\n"
# CHECK-NEXT: +sequence "\n"
# CHECK-NEXT:  terminates lines.
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "diff" "-u" "--strip-trailing-cr" "diff-in.unix" "diff-in.dos"
# CHECK: # command output:
# CHECK: @@
# CHECK-NEXT:  In this file, the
# CHECK-NEXT: -sequence "\n"
# CHECK-NEXT: +sequence "\r\n"
# CHECK-NEXT:  terminates lines.
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "false"

# CHECK: ***


# CHECK: FAIL: shtest-shell :: diff-unified.txt

# CHECK: *** TEST 'shtest-shell :: diff-unified.txt' FAILED ***

# CHECK: $ "diff" "-u" "{{[^"]*}}.foo" "{{[^"]*}}.bar"
# CHECK: # command output:
# CHECK: @@ {{.*}} @@
# CHECK-NEXT: 3
# CHECK-NEXT: 4
# CHECK-NEXT: 5
# CHECK-NEXT: -6 foo
# CHECK-NEXT: +6 bar
# CHECK-NEXT: 7
# CHECK-NEXT: 8
# CHECK-NEXT: 9
# CHECK-EMPTY:
# CHECK-NEXT: error: command failed with exit status: 1
# CHECK-NEXT: $ "true"

# CHECK: $ "diff" "-U" "2" "{{[^"]*}}.foo" "{{[^"]*}}.bar"
# CHECK: # command output:
# CHECK: @@ {{.*}} @@
# CHECK-NEXT: 4
# CHECK-NEXT: 5
# CHECK-NEXT: -6 foo
# CHECK-NEXT: +6 bar
# CHECK-NEXT: 7
# CHECK-NEXT: 8
# CHECK-EMPTY:
# CHECK-NEXT: error: command failed with exit status: 1
# CHECK-NEXT: $ "true"

# CHECK: $ "diff" "-U4" "{{[^"]*}}.foo" "{{[^"]*}}.bar"
# CHECK: # command output:
# CHECK: @@ {{.*}} @@
# CHECK-NEXT: 2
# CHECK-NEXT: 3
# CHECK-NEXT: 4
# CHECK-NEXT: 5
# CHECK-NEXT: -6 foo
# CHECK-NEXT: +6 bar
# CHECK-NEXT: 7
# CHECK-NEXT: 8
# CHECK-NEXT: 9
# CHECK-NEXT: 10
# CHECK-EMPTY:
# CHECK-NEXT: error: command failed with exit status: 1
# CHECK-NEXT: $ "true"

# CHECK: $ "diff" "-U0" "{{[^"]*}}.foo" "{{[^"]*}}.bar"
# CHECK: # command output:
# CHECK: @@ {{.*}} @@
# CHECK-NEXT: -6 foo
# CHECK-NEXT: +6 bar
# CHECK-EMPTY:
# CHECK-NEXT: error: command failed with exit status: 1
# CHECK-NEXT: $ "true"

# CHECK: $ "diff" "-U" "30.1" "{{[^"]*}}" "{{[^"]*}}"
# CHECK: # command stderr:
# CHECK: Error: invalid '-U' argument: 30.1
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "diff" "-U-1" "{{[^"]*}}" "{{[^"]*}}"
# CHECK: # command stderr:
# CHECK: Error: invalid '-U' argument: -1
# CHECK: error: command failed with exit status: 1
# CHECK: $ "true"

# CHECK: $ "false"

# CHECK: ***


# CHECK: FAIL: shtest-shell :: diff-w.txt
# CHECK: *** TEST 'shtest-shell :: diff-w.txt' FAILED ***
# CHECK: $ "diff" "-w" "{{[^"]*}}.0" "{{[^"]*}}.1"
# CHECK: # command output:
# CHECK: 1,3
# CHECK-NEXT: {{^  }}foo
# CHECK-NEXT: {{^  }}bar
# CHECK-NEXT: ! baz
# CHECK-NEXT: ---
# CHECK-NEXT: {{^  }}foo
# CHECK-NEXT: {{^  }}bar
# CHECK-NEXT: ! bat
# CHECK-EMPTY:
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: error-0.txt
# CHECK: *** TEST 'shtest-shell :: error-0.txt' FAILED ***
# CHECK: $ "not-a-real-command"
# CHECK: # command stderr:
# CHECK: 'not-a-real-command': command not found
# CHECK: error: command failed with exit status: 127
# CHECK: ***

# FIXME: The output here sucks.
#
# CHECK: FAIL: shtest-shell :: error-1.txt
# CHECK: *** TEST 'shtest-shell :: error-1.txt' FAILED ***
# CHECK: shell parser error on: ': \'RUN: at line 3\'; echo "missing quote'
# CHECK: ***

# CHECK: FAIL: shtest-shell :: error-2.txt
# CHECK: *** TEST 'shtest-shell :: error-2.txt' FAILED ***
# CHECK: Unsupported redirect:
# CHECK: ***

# CHECK: FAIL: shtest-shell :: mkdir-error-0.txt
# CHECK: *** TEST 'shtest-shell :: mkdir-error-0.txt' FAILED ***
# CHECK: $ "mkdir" "-p" "temp"
# CHECK: # command stderr:
# CHECK: Unsupported: 'mkdir' cannot be part of a pipeline
# CHECK: error: command failed with exit status: 127
# CHECK: ***

# CHECK: FAIL: shtest-shell :: mkdir-error-1.txt
# CHECK: *** TEST 'shtest-shell :: mkdir-error-1.txt' FAILED ***
# CHECK: $ "mkdir" "-p" "-m" "777" "temp"
# CHECK: # command stderr:
# CHECK: Unsupported: 'mkdir': option -m not recognized
# CHECK: error: command failed with exit status: 127
# CHECK: ***

# CHECK: FAIL: shtest-shell :: mkdir-error-2.txt
# CHECK: *** TEST 'shtest-shell :: mkdir-error-2.txt' FAILED ***
# CHECK: $ "mkdir" "-p"
# CHECK: # command stderr:
# CHECK: Error: 'mkdir' is missing an operand
# CHECK: error: command failed with exit status: 127
# CHECK: ***

# CHECK: PASS: shtest-shell :: redirects.txt

# CHECK: FAIL: shtest-shell :: rm-error-0.txt
# CHECK: *** TEST 'shtest-shell :: rm-error-0.txt' FAILED ***
# CHECK: $ "rm" "-rf" "temp"
# CHECK: # command stderr:
# CHECK: Unsupported: 'rm' cannot be part of a pipeline
# CHECK: error: command failed with exit status: 127
# CHECK: ***

# CHECK: FAIL: shtest-shell :: rm-error-1.txt
# CHECK: *** TEST 'shtest-shell :: rm-error-1.txt' FAILED ***
# CHECK: $ "rm" "-f" "-v" "temp"
# CHECK: # command stderr:
# CHECK: Unsupported: 'rm': option -v not recognized
# CHECK: error: command failed with exit status: 127
# CHECK: ***

# CHECK: FAIL: shtest-shell :: rm-error-2.txt
# CHECK: *** TEST 'shtest-shell :: rm-error-2.txt' FAILED ***
# CHECK: $ "rm" "-r" "hello"
# CHECK: # command stderr:
# CHECK: Error: 'rm' command failed
# CHECK: error: command failed with exit status: 1
# CHECK: ***

# CHECK: FAIL: shtest-shell :: rm-error-3.txt
# CHECK: *** TEST 'shtest-shell :: rm-error-3.txt' FAILED ***
# CHECK: Exit Code: 1
# CHECK: ***

# CHECK: PASS: shtest-shell :: rm-unicode-0.txt
# CHECK: PASS: shtest-shell :: sequencing-0.txt
# CHECK: XFAIL: shtest-shell :: sequencing-1.txt

# CHECK: FAIL: shtest-shell :: stdout-encoding.txt
# CHECK: *** TEST 'shtest-shell :: stdout-encoding.txt' FAILED ***
# CHECK: $ "cat" "diff-in.bin"
# CHECK: # command output:
# CHECK-NEXT: {{^.f.o.o.$}}
# CHECK-NEXT: {{^.b.a.r..}}
# CHECK-NEXT: {{^.b.a.z.$}}
# CHECK-NOT: error
# CHECK: $ "false"
# CHECK: ***

# CHECK: PASS: shtest-shell :: valid-shell.txt
# CHECK: Failed Tests (35)
